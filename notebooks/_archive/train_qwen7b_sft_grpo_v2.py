"""
Qwen2.5-7B SFT + GRPO on the Hospitality Env (v2, hardened against Colab quirks).

This is the "all-known-landmines-defused" version. Every pip install and
monkey-patch documented in `scratch/study_notes/03_dev_notes.md` §负一 and
§负零点五 is baked into Cell 1, so there are no late surprises.

Prereqs:
  - A100 GPU (Runtime → Change runtime type → A100)
  - Pay-As-You-Go compute units active
  - Upload `hospitality_sft.jsonl` to /content/ via left sidebar Files panel

Cell execution order (strict):
  1. Cell 1: install everything (~3-5 min)
  2. Runtime → Restart session
  3. Cell 2 onwards, top to bottom
"""

# %% [markdown]
# # Qwen2.5-7B SFT + GRPO on the Hospitality Env (v2)
# **Reminder:** after Cell 1 finishes, Runtime → Restart session, then run Cell 2 onwards.

# %% [markdown]
# ## Cell 1 — Install everything (then Restart session)

# %%
# Core stack
# !pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q "trl==0.15.2" "transformers>=4.51.3,<=5.5.0" accelerate bitsandbytes peft
# !pip install -q openenv-core==0.2.3 fastapi "uvicorn[standard]" httpx nest_asyncio
# !pip install -q --upgrade "datasets>=3.0"
#
# # All trl 0.15.2 secret lazy-import dependencies (discovered one-by-one in §负一)
# !pip install -q --no-deps mergekit llm-blender immutables dataclasses_json marshmallow typing_inspect mypy_extensions
#
# # Pin pydantic to the version that satisfies mergekit but stays under 2.12's strict checks
# !pip install -q --force-reinstall --no-deps "pydantic==2.10.6" "pydantic-core==2.27.2"
#
# print("✅ Cell 1 done. NOW: Runtime → Restart session, then continue with Cell 2.")

# %% [markdown]
# ## Cell 2 — Post-restart: monkey-patches BEFORE any trl import
#
# These must run first. `trl.__init__.py` calls `find_spec("vllm")` at module
# level and needs the stub in place; `llm_blender` references
# `transformers.utils.hub.TRANSFORMERS_CACHE` which was removed in 4.51+.

# %%
# (a) Patch TRANSFORMERS_CACHE back onto transformers.utils.hub
import os
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    _hub.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# (b) Install a proper vllm stub with __spec__ so find_spec() doesn't raise
import sys, types, importlib.machinery

def _make_stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__file__ = f"<stub:{name}>"
    m.__path__ = []
    return m

_vllm = _make_stub("vllm")
_vllm.LLM = type("LLM", (), {})
_vllm.SamplingParams = type("SamplingParams", (), {})
_vllm.PoolingParams = type("PoolingParams", (), {})
_vllm.RequestOutput = type("RequestOutput", (), {})
sys.modules["vllm"] = _vllm
for sub in ["vllm.config", "vllm.distributed", "vllm.distributed.parallel_state",
            "vllm.engine", "vllm.engine.arg_utils",
            "vllm.worker", "vllm.worker.worker"]:
    sys.modules[sub] = _make_stub(sub)

# (c) Drop any cached trl import so the patches above apply
for k in list(sys.modules):
    if k.startswith("trl"):
        del sys.modules[k]

print("✅ Monkey-patches applied. trl import should now work.")

# %% [markdown]
# ## Cell 3 — Install the env package

# %%
# !git clone https://github.com/<your-username>/openenv-hospitality.git 2>/dev/null || echo "Already cloned"
# %cd /content/openenv-hospitality
# !pip install --no-deps -q -e hospitality_env/

import nest_asyncio; nest_asyncio.apply()

# Verify the one thing that keeps breaking after restart
import openenv.core  # noqa
print("✅ openenv import OK")

# %% [markdown]
# ## Cell 4 — Launch env server (with stdout capture, in case it dies)

# %%
import subprocess, time, httpx

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "hospitality_env.server.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
time.sleep(8)
if server_proc.poll() is not None:
    print("❌ Server DIED:\n", server_proc.stdout.read())
else:
    r = httpx.get("http://127.0.0.1:8000/health", timeout=2.0)
    print("✅ Server up:", r.json())

# %% [markdown]
# ## Cell 5 — Verify SFT JSONL is uploaded
#
# Upload `hospitality_sft.jsonl` via Colab left sidebar (📁 Files → upload)
# to `/content/hospitality_sft.jsonl`.

# %%
# !ls -la /content/hospitality_sft.jsonl
# !wc -l /content/hospitality_sft.jsonl

# %% [markdown]
# ## Cell 6 — Load Qwen2.5-7B (4-bit) + attach LoRA

# %%
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 8192   # our task prompt ~4313 tokens; leave room for completion

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
# ## Cell 7 — SFT on Claude high-reward trajectories

# %%
import json
from datasets import Dataset

SFT_PATH = "/content/hospitality_sft.jsonl"
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
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
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
# ## Cell 8 — Sanity check: is SFT producing valid JSON tool calls?

# %%
FastLanguageModel.for_inference(model)
sample_prompt = tokenizer.apply_chat_template(
    sft_rows[0]["messages"][:2], tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
FastLanguageModel.for_training(model)

# %% [markdown]
# ## Cell 9 — Build GRPO prompts + reward function

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
# ## Cell 10 — GRPO training from the SFT checkpoint

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
    max_steps=50,
    logging_steps=1,
    save_steps=25,
    report_to="none",
    bf16=True,
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
# ## Cell 11 — Save adapter + eval on held-out tasks

# %%
OUTPUT_DIR = "./qwen7b_hospitality_lora_sft_grpo"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional push to Hub:
# from huggingface_hub import HfApi, login; login()
# HfApi().upload_folder(folder_path=OUTPUT_DIR,
#     repo_id="binleiwang/qwen2.5-7b-hospitality-sft-grpo", repo_type="model")

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
# ## Cell 12 — Cleanup

# %%
server_proc.terminate(); server_proc.wait(timeout=10)
print("Done.")
