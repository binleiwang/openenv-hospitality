"""
Train Qwen2.5-7B-Instruct on the Hospitality Env via SFT warm-start + GRPO.

v2 of the training notebook. Differences from `train_qwen_grpo.py`:
  - Base model:     Qwen2.5-1.5B → Qwen2.5-7B-Instruct-bnb-4bit
  - max_seq_length: 4096 → 8192 (our task prompt is 4313 tokens)
  - Pipeline:       GRPO only → SFT on Claude trajectories, then GRPO
  - Hardware:       T4 → A100 (Colab Pay-As-You-Go, ~12 units/hr)

Why SFT warm-start: base Qwen-7B does not reliably emit valid JSON
tool-call responses for this env out-of-the-box. GRPO's group-relative
advantage is zero when all K rollouts hit the same reward floor, so
training never moves. We first SFT on ~30-40 high-reward Claude traces
to give the policy a starting distribution that can produce diverse,
structurally valid outputs — then GRPO can meaningfully optimise.

Prereqs:
  - SFT JSONL uploaded to Colab (/content/hospitality_sft.jsonl),
    produced by `build_sft_dataset.py` from a baseline run with
    --save-transcripts.
  - Colab A100 runtime (Runtime → Change runtime type → A100).
  - Pay-As-You-Go compute units active.

Naming convention:
  - Formal: Hospitality RL Environment: A Hot Pot Restaurant Simulation
  - Short:  Hospitality Env
  - Code:   hospitality_env / HospitalityEnv
"""

# %% [markdown]
# # Qwen2.5-7B SFT + GRPO on the Hospitality Env
#
# End-to-end fine-tuning on a single A100. Pipeline:
# 1. Install stable dependency set (from Colab dependency-hell playbook)
# 2. Launch the hospitality_env FastAPI server
# 3. Load Qwen2.5-7B-Instruct (Unsloth 4-bit) with LoRA r=16
# 4. SFT 1 epoch on high-reward Claude trajectories
# 5. GRPO 50 steps from the SFT checkpoint
# 6. Eval on held-out tasks

# %% [markdown]
# ## 1. Install dependencies
#
# Exact versions come from `scratch/study_notes/03_dev_notes.md` §负一
# (Colab GRPO dependency-hell log). Do NOT change without reading that note.

# %%
# !pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q "trl==0.15.2" "transformers>=4.51.3,<=5.5.0" accelerate bitsandbytes
# !pip install -q "pydantic>=2.10,<2.12"
# !pip install -q openenv-core==0.2.3
# !pip install -q fastapi "uvicorn[standard]" httpx nest_asyncio
#
# NOTE: After pip install, do Runtime → Restart session before running the
# rest of this notebook. transformers/pydantic versions won't otherwise reload.

# %%
# Force pydantic back to 2.10.6 if anything upgraded it to 2.12+ after restart.
# !pip install --force-reinstall --no-deps "pydantic==2.10.6" "pydantic-core==2.27.2"

# %%
import nest_asyncio
nest_asyncio.apply()

# %%
# Clone env repo and install without deps (prevents pydantic re-upgrade via
# openenv-core → gradio → fastmcp chain).
# !git clone https://github.com/<your-username>/openenv-hospitality.git
# %cd openenv-hospitality
# !pip install --no-deps -e hospitality_env/

# %%
# Upload the SFT JSONL manually via Colab Files panel to /content/hospitality_sft.jsonl
# Or download from HF Hub if you pushed it:
# !huggingface-cli download <your-user>/hospitality-sft --local-dir /content/sft_data --repo-type dataset

# %% [markdown]
# ## 2. Launch the env server

# %%
import subprocess, time, httpx

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "hospitality_env.server.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
for _ in range(30):
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
        if r.status_code == 200:
            print("Server up:", r.json()); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Server failed to start")

# %% [markdown]
# ## 3. Load Qwen2.5-7B with max_seq_length=8192

# %%
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 8192   # our task prompt is ~4313 tokens; leave room for completion

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model.print_trainable_parameters()

# %% [markdown]
# ## 4. SFT warm-start on Claude high-reward trajectories
#
# Dataset: one JSONL line per task, schema:
#   {"messages": [{"role": "system"/"user"/"assistant", "content": ...}, ...],
#    "task_id": "...", "reward": 0.83, "turns": 5}
#
# We train only on the assistant turns (standard instruction-tuning masking).

# %%
import json
from datasets import Dataset

SFT_PATH = "/content/hospitality_sft.jsonl"   # or openenv-hospitality/sft_data/...
sft_rows = [json.loads(line) for line in open(SFT_PATH)]
print(f"SFT examples: {len(sft_rows)}")

def _format(row):
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

sft_ds = Dataset.from_list(sft_rows).map(_format, remove_columns=["messages"])

# %%
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./qwen7b_hospitality_sft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,       # LoRA SFT standard
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    save_strategy="no",
    bf16=True,                # A100 supports bf16
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=sft_ds,
)
sft_trainer.train()

# %% [markdown]
# Sanity check: generate on one task and verify we get a valid-looking JSON action.

# %%
FastLanguageModel.for_inference(model)
sample_prompt = tokenizer.apply_chat_template(
    sft_rows[0]["messages"][:2],   # system + first user
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

# Switch back to training mode for GRPO.
FastLanguageModel.for_training(model)

# %% [markdown]
# ## 5. Build GRPO prompts + reward function

# %%
import asyncio
from pathlib import Path
from hospitality_env.client import HospitalityEnv
from hospitality_env.models import HospitalityAction

TASKS_PATH = Path("hospitality_env/server/data/tasks.json")
ALL_TASKS = json.load(open(TASKS_PATH))
TRAIN_IDS = [t["id"] for t in ALL_TASKS[:100]]
EVAL_IDS  = [t["id"] for t in ALL_TASKS[100:]]

def build_prompt(sys_msg, cust_msg):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":sys_msg},
         {"role":"user","content":cust_msg}],
        tokenize=False, add_generation_prompt=True,
    )

async def _reset(tid):
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    try:
        obs = await client.reset(task_id=tid)
        return obs.observation
    finally:
        await client.close()

rows = []
for tid in TRAIN_IDS:
    o = asyncio.run(_reset(tid))
    rows.append({"task_id": tid,
                 "prompt": build_prompt(o.system_message, o.customer_message)})
train_dataset = Dataset.from_list(rows)
print(f"GRPO train: {len(train_dataset)}")

# %%
def reward_fn(prompts, completions, task_id=None, **kwargs):
    rewards = []
    for tid, comp in zip(task_id, completions):
        async def _score():
            client = HospitalityEnv(base_url="http://127.0.0.1:8000")
            try:
                await client.reset(task_id=tid)
                step = await client.step(HospitalityAction(message=comp))
                r = step.reward or 0.0
                for _ in range(3):
                    if step.done: break
                    step = await client.step(HospitalityAction(message="Goodbye."))
                    r += step.reward or 0.0
                return r
            finally:
                await client.close()
        rewards.append(asyncio.run(_score()))
    return rewards

# %% [markdown]
# ## 6. GRPO training from the SFT checkpoint

# %%
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="./qwen7b_hospitality_grpo",
    num_generations=4,
    max_prompt_length=4500,
    max_completion_length=1024,
    temperature=0.9,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_steps=50,             # A100 ~2 min/step × 50 ≈ 100 min
    logging_steps=1,
    save_steps=25,
    report_to="none",
    bf16=True,                # A100
    beta=0.04,
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
    args=grpo_config,
    train_dataset=train_dataset,
)
grpo_trainer.train()

# %% [markdown]
# ## 7. Save adapter + eval on held-out tasks

# %%
OUTPUT_DIR = "./qwen7b_hospitality_lora_sft_grpo"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: push to HF Hub.
# from huggingface_hub import HfApi, login
# login()
# HfApi().upload_folder(folder_path=OUTPUT_DIR,
#     repo_id="<your-user>/qwen2.5-7b-hospitality-sft-grpo",
#     repo_type="model",
#     commit_message="Qwen-7B SFT warm-start + GRPO, r=16")

# %%
FastLanguageModel.for_inference(model)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

async def rollout(tid):
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    try:
        obs = await client.reset(task_id=tid)
        prompt = build_prompt(obs.observation.system_message,
                              obs.observation.customer_message)
        comp = generate_text(prompt)
        step = await client.step(HospitalityAction(message=comp))
        r = step.reward or 0.0
        for _ in range(3):
            if step.done: break
            step = await client.step(HospitalityAction(message="Goodbye."))
            r += step.reward or 0.0
        return r
    finally:
        await client.close()

eval_rewards = [asyncio.run(rollout(tid)) for tid in EVAL_IDS]
import statistics as st
print(f"Eval mean:   {st.mean(eval_rewards):.3f}")
print(f"Eval median: {st.median(eval_rewards):.3f}")

# %% [markdown]
# ## 8. Cleanup

# %%
server_proc.terminate(); server_proc.wait(timeout=10)
print("Done.")
