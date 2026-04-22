"""
Train Qwen2.5-7B-Instruct on the Hospitality Env — production pipeline.

This notebook is the production pipeline that produced the final submission
model. Two earlier GRPO-based attempts live in `notebooks/_archive/` and are
summarized in the appendix at the bottom of this file:

  Phase 1 (Qwen2.5-1.5B + GRPO on T4)
      → intra-group reward collapse (reward_std=0 for 11 consecutive steps).
        Root cause: max_seq_length=4096 truncated the 4313-token prompt.

  Phase 2 (Qwen2.5-7B + SFT 1 epoch + GRPO on A100)
      → SFT sanity-check emitted valid JSON, but GRPO hit reward_std=0 again
        and the env server timed out at step 30. The initial multi-turn eval
        of the SFT checkpoint produced a *bimodal* reward distribution
        (mean 0.400: 7 tasks at reward>0, 9 tasks at reward=0) — the zero
        half was the model looping in apology mode without calling tools.

This pipeline replaces GRPO with two prompt-level interventions that
directly target the bimodal failure:

  (a) SFT 3 epochs (vs 1 in Phase 2) to firmly anchor the JSON output format
  (b) An anti-loop rule in the system prompt + a mid-rollout "no_tool_streak"
      nudge when the model goes 2+ turns without a tool call

Final eval on 15 held-out tasks (tid 100-115, one WebSocket error excluded):
    Mean    reward:  +0.659   (vs Phase 2 0.400, Claude Sonnet 4.5 baseline 0.604)
    Median  reward:  +0.663
    Min/Max reward:  -0.300 / +1.950
    Mean    turns :   4.06    (vs Phase 2 6.19, Claude baseline 6.72)
    Tasks r>0: 14/15;   Tasks r≤0: 1/15   (distribution tightened, no more bimodal)

Prereqs:
  - Colab A100 runtime (Runtime → Change runtime type → A100)
  - Google Drive mounted (Section 1b prompts for OAuth on first run)
    Recommended layout under /content/drive/MyDrive/hospitality/ :
      ├── hospitality_sft.jsonl    ← SFT dataset (upload once, ~150 KB)
      ├── checkpoints/sft/         ← per-epoch SFT checkpoints (auto-created)
      ├── qwen7b_hospitality_lora/ ← final adapter + tokenizer (auto-created)
      └── eval/eval_heldout.json   ← eval record (auto-created)
    The SFT JSONL is produced by `build_sft_dataset.py` from a baseline_eval
    run with --save-transcripts and --min-reward 0.7 (34 trajectories).
  - Pay-As-You-Go compute units active (~$3 total for full run)
  - Reproducibility: SEED=42 throughout; training loss curve should match
    2.66 → 0.85 → 0.24 across steps 1/8/15 within rounding.

Naming convention:
  - Formal: Hospitality RL Environment: A Hot Pot Restaurant Simulation
  - Short:  Hospitality Env
  - Code:   hospitality_env / HospitalityEnv
"""

# %% [markdown]
# # Qwen2.5-7B SFT + Anti-Loop Prompt on the Hospitality Env
#
# Pipeline:
#  1. Install a stable dependency set (see appendix at the bottom for the full patch log)
#  2. Apply stubs (vllm, TRANSFORMERS_CACHE monkey-patch) so Unsloth/trl imports work
#  3. Launch the hospitality_env FastAPI server
#  4. Load Qwen2.5-7B-Instruct (Unsloth 4-bit) with LoRA r=16, max_seq_length=8192
#  5. SFT 3 epochs on 34 high-reward Claude Sonnet 4.5 trajectories
#  6. Inline the agent wrapper (SYSTEM_PROMPT_V2 with anti-loop rule + multi-turn rollout)
#  7. Eval 16 held-out tasks (max_turns=8, no_tool_streak nudge)
#  8. Save adapter + push to HF Hub
#
# Appendix at the bottom documents what *didn't* work (GRPO attempts) so the
# lessons aren't lost.

# %% [markdown]
# ## 1. Install dependencies
#
# All versions are pinned for a reason — every pin corresponds to a
# specific error encountered during development. See the appendix at the
# bottom of this notebook for the full patch log and what each one guards
# against. Do not change versions without reading that appendix first.

# %%
# !pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q "trl==0.15.2" "transformers>=4.51.3,<=5.5.0" accelerate bitsandbytes
# !pip install -q "pydantic==2.10.6" "pydantic-core==2.27.2"   # exact pin; see appendix A.5
# !pip install -q "datasets>=3.0"                              # pyarrow ABI: must be >=3.0 on Colab
# !pip install -q openenv-core==0.2.3
# !pip install -q fastapi "uvicorn[standard]" httpx nest_asyncio
#
# # trl 0.15.2 lazy-import chain. All installed with --no-deps on purpose
# # to avoid dragging conflicting transitive versions into the graph.
# !pip install --no-deps -q mergekit llm_blender immutables dataclasses_json marshmallow typing_inspect mypy_extensions

# %%
# NOTE: After pip install finishes, Runtime → Restart session before running the
# rest of the notebook. transformers/pydantic versions will not reload otherwise.

# %%
# Force pydantic back to 2.10.6 if anything silently upgraded it to 2.12+.
# !pip install --force-reinstall --no-deps "pydantic==2.10.6" "pydantic-core==2.27.2"

# %%
import nest_asyncio
nest_asyncio.apply()

# %%
# Clone the env repo and install without deps (prevents pydantic re-upgrade via
# openenv-core → gradio → fastmcp transitive chain).
# !git clone https://github.com/binleiwang/openenv-hospitality.git
# %cd openenv-hospitality
# !pip install --no-deps -e hospitality_env/

# %% [markdown]
# ## 1b. Mount Google Drive (persistence across Colab disconnects)
#
# Colab runtimes disconnect after ~90 min of inactivity or ~12 hours of
# use. If the SFT checkpoint + eval JSON are written to /content/, they
# vanish with the runtime. Mount Drive once at the start so every artifact
# the notebook produces lands under `/content/drive/MyDrive/hospitality/`
# and survives disconnects.
#
# After mount, Colab will prompt for OAuth once per runtime.

# %%
import os
from google.colab import drive

drive.mount("/content/drive")

DRIVE_BASE = "/content/drive/MyDrive/hospitality"
os.makedirs(DRIVE_BASE, exist_ok=True)
os.makedirs(f"{DRIVE_BASE}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_BASE}/eval", exist_ok=True)
print(f"Drive base: {DRIVE_BASE}")

# Convention used throughout this notebook:
#   {DRIVE_BASE}/hospitality_sft.jsonl    ← SFT dataset (upload once)
#   {DRIVE_BASE}/checkpoints/...          ← SFT per-epoch checkpoints
#   {DRIVE_BASE}/qwen7b_hospitality_lora  ← final adapter + tokenizer
#   {DRIVE_BASE}/eval/eval_heldout.json   ← eval results

# %% [markdown]
# ## 2. Stubs: TRANSFORMERS_CACHE + vllm
#
# Two monkey-patches that must happen *before* any `import trl` or
# `import unsloth`, or imports fail with cryptic errors.

# %%
import os, sys, types, importlib.machinery, pathlib

# 2a. trl 0.15.2 imports `transformers.TRANSFORMERS_CACHE` which was removed
#     in transformers 4.51. Restore it.
import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# 2b. Unsloth does a three-stage check for vllm:
#       (i)  sys.modules["vllm"]
#       (ii) vllm.__spec__ is a real ModuleSpec
#       (iii) importlib.metadata.version("vllm") returns a pip-registered version
#     A dict stub only satisfies (i). Write a tiny pip-installable stub to
#     satisfy all three.
STUB_DIR = "/tmp/vllm_stub"
if not os.path.exists(STUB_DIR):
    pathlib.Path(f"{STUB_DIR}/vllm").mkdir(parents=True, exist_ok=True)
    for sub in ["sampling_params", "config", "distributed", "engine", "worker"]:
        pathlib.Path(f"{STUB_DIR}/vllm/{sub}").mkdir(exist_ok=True)
        open(f"{STUB_DIR}/vllm/{sub}/__init__.py", "w").write(
            "class GuidedDecodingParams:\n    pass\n" if sub == "sampling_params" else ""
        )
    open(f"{STUB_DIR}/vllm/__init__.py", "w").write("__version__ = '0.6.3'\n")
    open(f"{STUB_DIR}/pyproject.toml", "w").write(
        "[project]\nname = 'vllm'\nversion = '0.6.3'\n"
    )
    # !pip install -q -e /tmp/vllm_stub

# 2c. trl 0.15.2's GRPOTrainer._get_train_sampler has a signature mismatch
#     with transformers 4.51+ (one arg vs two). We don't use GRPO in this
#     pipeline, but the fix is kept in the archive notebooks for reference.

# %% [markdown]
# ## 3. Launch the env server
#
# IMPORTANT: pipe stderr. "stdout=DEVNULL, stderr=DEVNULL" silently hides
# the error when the server fails to start. Cost me ~2 hours on Phase 2.

# %%
import subprocess, time, httpx

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "hospitality_env.server.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,  # NOT DEVNULL — see appendix diagnostic #4
)
for _ in range(30):
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
        if r.status_code == 200:
            print("Server up:", r.json()); break
    except Exception:
        time.sleep(1)
else:
    err = server_proc.stderr.read().decode(errors="ignore")[-2000:]
    raise RuntimeError(f"Server failed. stderr tail:\n{err}")

# %% [markdown]
# ## 4. Reproducibility + load Qwen2.5-7B-Instruct with LoRA
#
# max_seq_length=8192 because the env's task prompt is ~4313 tokens (system
# message + policy excerpt + tool schemas) and we need headroom for a
# multi-turn conversation history inside the same sequence.
#
# SEED=42 is set before model load AND passed into SFTConfig below. Note:
# under bf16 + 4-bit quantization, seed fixes the *training* trajectory
# (same loss curve on re-run), but greedy inference (do_sample=False) on
# the same prompt can still drift by <1% due to non-deterministic kernel
# ordering. Eval mean from a fresh run should land within ±0.02 of 0.659.

# %%
from unsloth import FastLanguageModel
from transformers import set_seed
import torch, random, numpy as np

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MAX_SEQ_LENGTH = 8192

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model.print_trainable_parameters()

# %% [markdown]
# ## 5. SFT warm-start on Claude Sonnet 4.5 high-reward trajectories
#
# Dataset: 34 trajectories from the v3-rerun baseline (reward ≥ 0.7),
# mean 7.3 turns/trajectory. See `build_sft_dataset.py` for the filter.

# %%
import json
from datasets import Dataset

# Look in Drive first (persists across sessions), fall back to /content/.
SFT_PATH = f"{DRIVE_BASE}/hospitality_sft.jsonl"
if not os.path.exists(SFT_PATH):
    SFT_PATH = "/content/hospitality_sft.jsonl"
print(f"SFT dataset: {SFT_PATH}")

sft_rows = [json.loads(line) for line in open(SFT_PATH)]
print(f"SFT examples: {len(sft_rows)}")
print(f"Mean turns:   {sum(r['turns'] for r in sft_rows) / len(sft_rows):.1f}")
print(f"Mean reward:  {sum(r['reward'] for r in sft_rows) / len(sft_rows):.2f}")

def _format(row):
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

sft_ds = Dataset.from_list(sft_rows).map(_format, remove_columns=["messages"])

# %%
# Sanity check #1: make sure every SFT example fits in max_seq_length.
# Phase 1's GRPO died because Unsloth silently truncated a 4313-token
# prompt at max_seq_length=4096. Catch that class of bug here.
_lens = [len(tokenizer(r["text"]).input_ids) for r in sft_ds]
print(f"Prompt token lengths: min={min(_lens)}, "
      f"mean={sum(_lens)/len(_lens):.0f}, max={max(_lens)}")
assert max(_lens) < MAX_SEQ_LENGTH, (
    f"Longest SFT example is {max(_lens)} tokens, which is >= "
    f"max_seq_length={MAX_SEQ_LENGTH}. Raise MAX_SEQ_LENGTH or trim data."
)
print(f"OK — longest example {max(_lens)} < max_seq_length {MAX_SEQ_LENGTH}")

# %%
from trl import SFTTrainer, SFTConfig

# NOTE: `max_length` was removed from SFTConfig in our trl/transformers combo.
# Do NOT pass it — it will raise TypeError at init. Default packing behavior
# on our 8192-seq model is what we want anyway.
sft_config = SFTConfig(
    output_dir=f"{DRIVE_BASE}/checkpoints/sft",   # Drive — survives disconnects
    num_train_epochs=3,                 # 3 epochs (Phase 2 used 1, underfit)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,                 # LoRA SFT standard
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    save_strategy="epoch",              # save each epoch (Phase 2 used "no", lost work on GRPO crash)
    bf16=True,                          # A100
    dataset_text_field="text",
    packing=False,
    report_to="none",
    seed=SEED,                          # reproducibility
    data_seed=SEED,                     # deterministic shuffle order
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=sft_ds,
)
sft_trainer.train()

# Expected loss curve (actual run):
#   step 1  → 2.66
#   step 8  → 0.85
#   step 15 → 0.24
# If step 15 loss > 0.5, SFT did not converge — check tokenizer template
# and max_seq_length before proceeding.

# %% [markdown]
# ### SFT sanity check
#
# Before running the full eval, generate once on a training example to
# confirm the model now emits valid JSON.

# %%
FastLanguageModel.for_inference(model)

sample_prompt = tokenizer.apply_chat_template(
    sft_rows[0]["messages"][:2],   # system + first user
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
with __import__("torch").no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

# Expected: a valid JSON action of the form
#   {"message": "...", "tool_name": "...", "tool_args": {...}}
# If you see free-form English instead, the prompt template used at inference
# does not match the template used during SFT. See §6 below — this is exactly
# the bug that caused Phase 2's bimodal eval (mean=0.400).

# %% [markdown]
# ## 6. Agent wrapper (inlined from `agent_utils.py`)
#
# This is the key Phase 2 → Phase 3 fix. Earlier versions of the eval used
# only `obs.system_message` + `obs.customer_message` as the prompt, which is
# a *different distribution* from what the model was trained on — the
# training prompts (from baseline_eval.py) used a rich template with task
# context, tool schemas, and turn numbers. Feeding the model a naked
# observation at inference produced English prose instead of JSON, and
# eval collapsed.
#
# Two additions beyond the original `agent_utils.py`:
#   (1) SYSTEM_PROMPT_V2 adds an explicit anti-loop rule
#   (2) rollout_v3 injects a [SYSTEM NUDGE] if the model goes 2+ turns
#       without calling a tool
#
# Together these remove the "loop in apology mode" failure that caused the
# 9 zero-reward tasks in Phase 2's eval.

# %%
from hospitality_env import HospitalityAction
import json as _json

SYSTEM_PROMPT_V2 = """You are a customer service agent (server) at Berkeley Hot Pot restaurant.
You MUST follow the staff policy given in the task system_message.

At each turn you receive the customer's last message and any tool results.
You can respond with ONE of three actions, expressed as strict JSON:

  {"message": "<text to customer>", "tool_name": "", "tool_args": {}}
  {"message": "", "tool_name": "<tool>", "tool_args": {...}}
  {"message": "<text>", "tool_name": "<tool>", "tool_args": {...}}

Rules:
- Output ONLY the JSON object, no markdown fences, no commentary.
- Use tools before making factual claims (allergies, order details, policy limits).
- Escalate per policy when required (SEVERE incidents, authority limits).
- Keep messages concise and empathetic. Avoid small talk or filler.

CRITICAL anti-loop rule:
- If your previous turn already apologized or expressed empathy,
  the NEXT turn MUST have a non-empty tool_name. Do not apologize twice in
  a row without taking action. Empathy is one turn; action is the next.

Closing conversations (important — avoid runaway chats):
- Once you have provided an answer or solution, briefly confirm it meets the
  customer's need.
- If the customer seems satisfied, ask "Is there anything else?" exactly once.
- If they say no, send a warm brief closing and stop.
- A task completed in 4 turns is better than the same task in 12 turns.
"""


def obs_to_dict(obs) -> dict:
    return {
        "customer_message": obs.customer_message,
        "tool_result": obs.tool_result,
        "tool_error": obs.tool_error,
        "system_message": obs.system_message,
        "available_tools": obs.available_tools,
        "tool_schemas": obs.tool_schemas,
        "task_description": obs.task_description,
        "turn_number": obs.turn_number,
        "max_turns": obs.max_turns,
        "metadata": obs.metadata,
    }


def build_user_turn(obs: dict, first_turn: bool) -> str:
    parts = []
    if first_turn and obs.get("system_message"):
        parts.append(f"=== TASK CONTEXT ===\n{obs['system_message']}")
        if obs.get("tool_schemas"):
            parts.append(
                f"\n=== TOOL SCHEMAS ===\n{_json.dumps(obs['tool_schemas'], indent=2)}"
            )
    if obs.get("customer_message"):
        parts.append(f"\n=== CUSTOMER SAYS ===\n{obs['customer_message']}")
    if obs.get("tool_result"):
        parts.append(f"\n=== LAST TOOL RESULT ===\n{obs['tool_result']}")
    if obs.get("tool_error"):
        parts.append(f"\n=== LAST TOOL ERROR ===\n{obs['tool_error']}")
    parts.append(
        f"\n[turn {obs.get('turn_number', 0)}/{obs.get('max_turns', 20)}] "
        f"Respond with JSON."
    )
    return "\n".join(parts)


def parse_action(text: str) -> HospitalityAction:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return HospitalityAction(message=text[:500])
    try:
        data = _json.loads(text[start : end + 1])
    except _json.JSONDecodeError:
        return HospitalityAction(message=text[:500])
    return HospitalityAction(
        message=data.get("message", "") or "",
        tool_name=data.get("tool_name", "") or "",
        tool_args=data.get("tool_args", {}) or {},
    )

# %% [markdown]
# ## 7. Multi-turn rollout eval
#
# max_turns=8 is deliberate. The env's own close-logic fallback triggers at
# turn 8 (substantive reply + ≥1 tool call → done), and earlier eval attempts
# with max_turns=20 hit context overflow (message accumulation >10k tokens
# after turn 12). Align rollout horizon with env close horizon.

# %%
import asyncio, torch, statistics as st
from pathlib import Path
from hospitality_env.client import HospitalityEnv

TASKS_PATH = Path("hospitality_env/server/data/tasks.json")
ALL_TASKS = json.load(open(TASKS_PATH))
EVAL_IDS = [t["id"] for t in ALL_TASKS[100:]]   # held-out (trained on [0:100])

def generate_text(messages, max_new_tokens=384):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LENGTH - max_new_tokens).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


async def rollout_v3(tid: str, max_turns: int = 8, verbose: bool = False):
    """Multi-turn rollout with anti-loop nudge."""
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    total_reward, turns, tool_calls = 0.0, 0, 0
    no_tool_streak = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2}]
    try:
        step = await client.reset(task_id=tid)
        obs = step.observation
        for turn in range(max_turns):
            user_content = build_user_turn(obs_to_dict(obs), first_turn=(turn == 0))
            if no_tool_streak >= 2:
                user_content += (
                    f"\n\n[SYSTEM NUDGE] You have not called a tool in "
                    f"{no_tool_streak} turns. If this ticket requires a lookup, "
                    f"comp, or escalation, take that action now via tool_name."
                )
            messages.append({"role": "user", "content": user_content})

            raw = generate_text(messages)
            action = parse_action(raw)
            messages.append({"role": "assistant", "content": raw})

            if action.tool_name:
                tool_calls += 1; no_tool_streak = 0
            else:
                no_tool_streak += 1

            step = await client.step(action)
            total_reward += step.reward or 0.0
            turns += 1
            if verbose:
                print(f"  turn {turn}: tool={action.tool_name!r} r={step.reward:+.2f}")
            if step.done:
                break
            obs = step.observation
        return {"task_id": tid, "reward": total_reward, "turns": turns,
                "tool_calls": tool_calls}
    except Exception as e:
        return {"task_id": tid, "reward": None, "turns": turns,
                "tool_calls": tool_calls, "error": str(e)[:200]}
    finally:
        await client.close()


# %%
# Run the held-out eval. 16 tasks × ~1 min/task ≈ 16 minutes on A100.
results = []
for tid in EVAL_IDS:
    r = asyncio.run(rollout_v3(tid))
    results.append(r)
    if r.get("error"):
        print(f"{tid}: ERROR {r['error']}")
    else:
        print(f"{tid}: r={r['reward']:+.3f} turns={r['turns']} tools={r['tool_calls']}")

# Summary.
valid = [r for r in results if r["reward"] is not None]
rewards = [r["reward"] for r in valid]
print(f"\n=== Held-out eval ===")
print(f"Mean:   {st.mean(rewards):+.3f}")
print(f"Median: {st.median(rewards):+.3f}")
print(f"Min/Max: {min(rewards):+.3f} / {max(rewards):+.3f}")
print(f"Mean turns: {st.mean([r['turns'] for r in valid]):.2f}")
print(f"Claude Sonnet 4.5 baseline on same distribution: +0.604")

# Actual numbers from the shipped run (one WebSocket error on task 059, excluded from mean):
#   Mean:    +0.659
#   Median:  +0.663
#   Min/Max: -0.300 / +1.950
#   Mean turns: 4.06
#   Tasks r>0: 14/15   Tasks r≤0: 1/15

# %%
# Save the eval record for the blog — to Drive so it survives disconnects.
EVAL_OUT = f"{DRIVE_BASE}/eval/eval_heldout.json"
with open(EVAL_OUT, "w") as f:
    json.dump({"results": results,
               "summary": {"mean": st.mean(rewards),
                           "median": st.median(rewards),
                           "n": len(valid),
                           "seed": SEED}}, f, indent=2)
print(f"Wrote {EVAL_OUT}")

# %% [markdown]
# ## 8. Save the adapter + push to HF Hub

# %%
# Save to Drive first (survives disconnect) — optional second copy under /content/.
OUTPUT_DIR = f"{DRIVE_BASE}/qwen7b_hospitality_lora"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved adapter + tokenizer to {OUTPUT_DIR}")

# %%
# Push to HF Hub (optional).
# from huggingface_hub import HfApi, login
# login()
# HfApi().upload_folder(
#     folder_path=OUTPUT_DIR,
#     repo_id="binleiwang/qwen2.5-7b-hospitality-sft",
#     repo_type="model",
#     commit_message="SFT 3ep + anti-loop prompt, held-out eval 0.659 vs Claude 0.604",
# )

# %% [markdown]
# ## 9. (Optional) GRPO fine-tune from the SFT checkpoint
#
# **Skip this section if SFT + anti-loop prompt already gave you numbers
# you're happy with.** Set `RUN_GRPO = False` and jump to Section 10.
#
# GRPO failed twice on this env before (Phase 1: reward_std=0 from prompt
# truncation; Phase 2: reward_std=0 from SFT collapsing rollout diversity).
# If you want to try a third time, this section applies every lesson from
# those failures:
#
#   - **K = 8** (not 4). Doubles rollouts per prompt so the SFT-collapsed
#     distribution has a chance to produce varied completions.
#   - **temperature = 1.3** (not 0.9). Forces more diversity on top of SFT.
#   - **beta = 0.01** (not 0.04). Lower KL pressure so the adapter isn't
#     pulled back to base when advantage is zero.
#   - **max_steps = 20** (not 50). Cheaper per attempt; abort early if it's
#     not moving.
#   - **reward_std watchdog**: a TrainerCallback reads log_history each
#     step and raises if `reward_std == 0` for 5 consecutive steps.
#   - **_get_train_sampler signature patch**: trl 0.15.2 vs transformers
#     4.51 disagree on arity. Must be applied before GRPOTrainer runs.
#
# Cost: ~$1-2 in Colab compute units, ~30-40 min on A100.
# If it works, mean reward should move from 0.66 → 0.70+.
# If `reward_std=0` fires for 5 steps, training aborts and you keep the
# SFT checkpoint.

# %%
RUN_GRPO = False   # Flip to True to attempt GRPO. Read the block above first.

# %%
if RUN_GRPO:
    # Patch 18: trl 0.15.2's GRPOTrainer._get_train_sampler has a signature
    # mismatch with transformers 4.51+ (one arg vs two). Apply the monkey-
    # patch BEFORE constructing any GRPOTrainer instance.
    from trl import GRPOTrainer
    _orig_sampler = GRPOTrainer._get_train_sampler
    def _patched_sampler(self, *args, **kwargs):
        return _orig_sampler(self)
    GRPOTrainer._get_train_sampler = _patched_sampler
    print("Applied _get_train_sampler signature patch.")

# %%
if RUN_GRPO:
    # Build GRPO train dataset: one row per task, prompt = first observation.
    # GRPO will only optimize the first-turn action; that's fine because for
    # this env the first tool call + first message determines whether the
    # ticket closes in 1-3 turns.
    import asyncio
    from hospitality_env.client import HospitalityEnv

    async def _fetch_first_prompt(tid):
        client = HospitalityEnv(base_url="http://127.0.0.1:8000")
        try:
            step = await client.reset(task_id=tid)
            obs = step.observation
            user = build_user_turn(obs_to_dict(obs), first_turn=True)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": user}]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        finally:
            await client.close()

    TRAIN_IDS = [t["id"] for t in ALL_TASKS[:100]]   # first 100 tasks for training
    grpo_rows = []
    for tid in TRAIN_IDS:
        grpo_rows.append({
            "task_id": tid,
            "prompt": asyncio.run(_fetch_first_prompt(tid)),
        })
    grpo_ds = Dataset.from_list(grpo_rows)
    print(f"GRPO train dataset: {len(grpo_ds)} prompts")

    # Sanity check: largest GRPO prompt must also fit in max_seq_length,
    # with headroom for completion (max_completion_length below).
    _glen = [len(tokenizer(r["prompt"]).input_ids) for r in grpo_rows]
    print(f"GRPO prompt lengths: min={min(_glen)}, max={max(_glen)}")
    assert max(_glen) + 1024 < MAX_SEQ_LENGTH, (
        f"GRPO prompt {max(_glen)} + 1024 completion "
        f">= max_seq_length={MAX_SEQ_LENGTH}. Training will truncate."
    )

# %%
if RUN_GRPO:
    # Reward function: score a single model completion by stepping the env.
    from hospitality_env.models import HospitalityAction

    def grpo_reward_fn(prompts, completions, task_id=None, **kwargs):
        rewards = []
        for tid, comp in zip(task_id, completions):
            async def _score():
                client = HospitalityEnv(base_url="http://127.0.0.1:8000")
                try:
                    await client.reset(task_id=tid)
                    action = parse_action(comp)
                    step = await client.step(action)
                    r = step.reward or 0.0
                    # Let the env close out so it computes final task reward.
                    for _ in range(3):
                        if step.done: break
                        step = await client.step(
                            HospitalityAction(message="Thank you, goodbye.")
                        )
                        r += step.reward or 0.0
                    return r
                finally:
                    await client.close()
            rewards.append(asyncio.run(_score()))
        return rewards

# %%
if RUN_GRPO:
    # Watchdog callback: abort training if reward_std=0 for 5 steps in a row.
    from transformers import TrainerCallback

    class RewardStdWatchdog(TrainerCallback):
        def __init__(self, patience=5):
            self.patience = patience
            self.streak = 0

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs: return
            rstd = logs.get("reward_std", None)
            if rstd is None: return
            if rstd == 0.0:
                self.streak += 1
                print(f"  [watchdog] reward_std=0 streak: {self.streak}/{self.patience}")
                if self.streak >= self.patience:
                    print("  [watchdog] ABORT: reward collapse detected.")
                    control.should_training_stop = True
            else:
                if self.streak > 0:
                    print(f"  [watchdog] reward_std={rstd:.4f} — cleared streak")
                self.streak = 0

# %%
if RUN_GRPO:
    from trl import GRPOConfig

    grpo_config = GRPOConfig(
        output_dir=f"{DRIVE_BASE}/checkpoints/grpo",
        num_generations=8,               # K=8 (Phase 2 used 4, collapsed)
        max_prompt_length=5500,
        max_completion_length=1024,
        temperature=1.3,                 # high diversity (Phase 2 used 0.9)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=20,                    # small budget, abort early if bad
        logging_steps=1,                 # MANDATORY: else we can't watch reward_std
        save_steps=10,
        report_to="none",
        bf16=True,
        beta=0.01,                       # low KL pressure (Phase 2 used 0.04)
        seed=SEED,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=grpo_reward_fn,
        args=grpo_config,
        train_dataset=grpo_ds,
        callbacks=[RewardStdWatchdog(patience=5)],
    )
    grpo_trainer.train()

    # After training, dump log_history for blog evidence.
    import pprint
    print("\n=== last 5 log_history entries ===")
    pprint.pprint(grpo_trainer.state.log_history[-5:])

# %%
if RUN_GRPO:
    # Re-run held-out eval with the GRPO-updated weights.
    FastLanguageModel.for_inference(model)
    grpo_results = []
    for tid in EVAL_IDS:
        r = asyncio.run(rollout_v3(tid))
        grpo_results.append(r)
        if r.get("error"):
            print(f"{tid}: ERROR {r['error']}")
        else:
            print(f"{tid}: r={r['reward']:+.3f} turns={r['turns']} tools={r['tool_calls']}")

    gvalid = [r for r in grpo_results if r["reward"] is not None]
    grewards = [r["reward"] for r in gvalid]
    print(f"\n=== Post-GRPO held-out eval ===")
    print(f"Mean:   {st.mean(grewards):+.3f}  (SFT-only baseline: ~0.659)")
    print(f"Median: {st.median(grewards):+.3f}")
    print(f"Mean turns: {st.mean([r['turns'] for r in gvalid]):.2f}")

    GRPO_OUTPUT_DIR = f"{DRIVE_BASE}/qwen7b_hospitality_lora_sft_grpo"
    model.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
    print(f"Saved GRPO-updated adapter to {GRPO_OUTPUT_DIR}")

    with open(f"{DRIVE_BASE}/eval/eval_heldout_postgrpo.json", "w") as f:
        json.dump({"results": grpo_results,
                   "summary": {"mean": st.mean(grewards),
                               "median": st.median(grewards),
                               "n": len(gvalid),
                               "seed": SEED}}, f, indent=2)

# %% [markdown]
# ## 10. Cleanup

# %%
server_proc.terminate()
try: server_proc.wait(timeout=10)
except subprocess.TimeoutExpired: server_proc.kill()
print("Done.")

# %% [markdown]
# ---
#
# ## Appendix: what didn't work (and why it's in this notebook as a comment)
#
# This appendix is not executable — it's a record of the two GRPO attempts
# that came before this pipeline, so the lessons are preserved in the same
# file as the working pipeline.
#
# ### A1. GRPO on Qwen2.5-1.5B (T4)
#
# Symptom: `loss = 0.000000` for 11 consecutive steps. Trainer HTML table
# showed *only* the loss column, which masked the real issue. Reading
# `trainer.state.log_history` revealed:
#
#     reward = 0.12   reward_std = 0.0   kl = 0.0   grad_norm = 0.0
#
# For 11 steps in a row. This is **intra-group reward collapse**: all K
# rollouts for every prompt landed on the same reward floor, so the
# group-relative advantage `(r_i − mean) / (std + eps)` was zero, and GRPO
# has no signal to learn from.
#
# Root cause: `max_seq_length=4096` but the task prompt is ~4313 tokens, so
# Unsloth silently truncated the prompt. The 1.5B base model with a
# truncated prompt produced structurally invalid outputs that all hit the
# same reward floor. K identical rollouts → zero variance → zero advantage.
#
# Lesson 1: **when GRPO loss is exactly 0, read `log_history`, not the HTML
# table.** The triad to check is `reward_std`, `kl`, `grad_norm` — not loss
# magnitude.
#
# ### A2. GRPO on Qwen2.5-7B + SFT 1 epoch (A100)
#
# Symptom: `loss` wiggled in [0.001, 0.003] for 30 steps, then the env
# server crashed with `WebSocket CancelledError / keepalive ping timeout`.
# The wiggling loss *looked* healthy, but `log_history` showed:
#
#     reward = 0.58   reward_std = 0.0   kl ≠ 0   grad_norm ≠ 0
#
# The KL and grad_norm being non-zero came from the β×KL regularization
# term pulling the LoRA adapter back toward the base policy (since
# advantage was still zero). In other words: GRPO was training the model
# to **unlearn** the SFT, because SFT'd behavior diverges from the base
# model and the KL pressure is the only gradient signal when advantage=0.
#
# Root cause: K=4 rollouts with temperature=0.9 on the SFT'd policy were
# too similar to each other (SFT collapses output diversity by
# construction), so again reward_std ≈ 0.
#
# Lesson 2: **SFT warm-start + low K + low temperature is a GRPO death
# spiral.** If you SFT first, you need to either raise K (8+), raise
# temperature (1.2+), or lower β (0.01) — otherwise the KL term will drag
# the policy back toward base.
#
# ### A3. Bimodal Phase 2 eval (mean 0.400)
#
# Symptom: SFT 1-epoch checkpoint, evaluated on 16 tasks, produced:
#   7 tasks at reward > 0 (often 0.6-1.2)
#   9 tasks at reward = 0
# The zero-half all had the same failure mode: model apologized in English,
# never called a tool, eventually ran out the turn budget.
#
# Root cause: *prompt distribution mismatch.* The SFT dataset used the rich
# template from `agent_utils.build_user_turn` (task context, tool schemas,
# customer message, turn counter). The initial eval script used only
# `obs.system_message` + `obs.customer_message` — a *different* prompt
# distribution, and the 7B model under 4-bit quantization couldn't bridge
# the two.
#
# Diagnostic: feeding the model `sft_rows[0]["messages"][:2]` produced
# perfect JSON; feeding it `build_prompt(obs.system_message,
# obs.customer_message)` produced English prose. That confirmed the
# mismatch — the fix was inlining `agent_utils.py` into the eval loop.
#
# Lesson 3: **eval prompts must match training prompts character-for-
# character.** Especially under 4-bit quantization: the model has less
# room to generalize across surface-level prompt differences.
#
# ### A4. Why the shipped pipeline skips GRPO
#
# After A3 was fixed (with the rich prompt template + anti-loop rule + 3
# SFT epochs), eval mean moved from 0.400 → 0.659 on the same 16 held-out
# tasks. At that point, retrying GRPO was a worse bet than shipping:
#
#   - the remaining failures were *format-correct but strategically wrong*
#     tasks, not the bimodal "refuses to call a tool" failure
#   - GRPO with K=4 couldn't generate enough variance on an SFT'd 7B to
#     escape reward_std=0, and raising K to 8 would double the A100 cost
#   - P1 (prompt engineering) already beat the Claude Sonnet 4.5 baseline
#     (0.659 > 0.604), which is the headline result for the blog
#
# Decision: ship SFT + anti-loop prompt. Document GRPO attempts honestly
# in the blog's lessons section. Don't optimize a broken pipeline just
# because the plan said "GRPO."
#
# ---
#
# ## Appendix B: Full dependency-patch inventory
#
# Every pip pin and monkey-patch in this notebook corresponds to a specific
# error encountered during Phase 2 development. Listed roughly in the order
# you'll trip them:
#
# | # | Symptom | Root cause | Patch |
# |---|---------|-----------|-------|
# | A.1 | `pyarrow.lib.IpcReadOptions size changed` | datasets and pyarrow ABI mismatch on Colab | **Upgrade `datasets>=3.0`**. Do NOT downgrade pyarrow — Colab has two pyarrow versions side by side and downgrading is a rabbit hole. |
# | A.2 | `uvicorn` can't find `hospitality_env.server` | No `__main__` in the package | Use `python -m uvicorn hospitality_env.server.app:app --host 127.0.0.1 --port 8000` — the explicit app path form. |
# | A.3 | env server subprocess dies silently | `subprocess.DEVNULL` on stderr hides the real error | Always pipe stderr: `stderr=subprocess.PIPE`, then `proc.stderr.read().decode()` on failure. |
# | A.4 | `ModuleNotFoundError: No module named 'openenv'` in server process | Local dev uses an editable install with a different import name than the pip release | `pip install openenv-core==0.2.3`. Install AFTER cloning the repo with `--no-deps` to avoid pulling in gradio/fastmcp which upgrades pydantic. |
# | A.5 | Cryptic errors like `cannot import name 'PydanticDeprecatedSince211'` | Transitive deps upgrade pydantic to 2.12+, mergekit needs 2.10.6 | `pip install --force-reinstall --no-deps "pydantic==2.10.6" "pydantic-core==2.27.2"`. Jupyter memory-vs-disk version mismatch: restart runtime after this. |
# | A.6 | `SFTConfig.__init__() got unexpected kw 'max_length'` | trl 0.15.2 + transformers 4.51 combo removed `max_length` from SFTConfig | Just don't pass it. Default behavior on an 8192-seq model is what you want. |
# | A.7 | `ModuleNotFoundError: mergekit` when importing GRPOTrainer | trl 0.15.2 lazy-imports mergekit (model-merging extension) which nothing else in your env needs | `pip install --no-deps mergekit` — and the chain it drags in: `llm_blender immutables dataclasses_json marshmallow typing_inspect mypy_extensions`. All `--no-deps` to avoid version conflicts. |
# | A.8 | `AttributeError: TRANSFORMERS_CACHE` during trl import | Constant removed in transformers 4.51+, but trl 0.15.2 still imports it | Monkey-patch it back BEFORE `import trl`: `transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")`. |
# | A.9 | `No module named 'vllm'` or `vllm.__spec__ is None` or `PackageNotFoundError: vllm` | Unsloth does a three-stage check: sys.modules + __spec__ + pip metadata. A dict stub only satisfies stage 1. | Write a tiny pip-installable stub: `/tmp/vllm_stub/pyproject.toml` declaring `name=vllm version=0.6.3`, plus empty submodule directories for `vllm.sampling_params`, `vllm.config`, `vllm.distributed`, `vllm.engine`, `vllm.worker`. `pip install -e /tmp/vllm_stub`. Satisfies all three stages. |
# | A.10 | `_get_train_sampler() takes 1 positional argument but 2 were given` | trl 0.15.2 defines `_get_train_sampler(self)`; transformers 4.51+ calls it with `(self, dataset)` | Monkey-patch to wrap: `_orig = GRPOTrainer._get_train_sampler; GRPOTrainer._get_train_sampler = lambda self, *a, **kw: _orig(self)`. Apply before constructing the trainer. |
#
# ## Appendix C: GRPO diagnostic checklist
#
# If you turn `RUN_GRPO = True` above and it doesn't behave, these are the
# checks to run in order:
#
# 1. **Read `log_history`, not the HTML table.** The HF Trainer HTML
#    progress table only shows `Training Loss`. The real metrics
#    (`reward`, `reward_std`, `kl`, `grad_norm`) live in
#    `trainer.state.log_history`. Example:
#    ```python
#    import pprint
#    pprint.pprint(trainer.state.log_history[-5:])
#    ```
#
# 2. **The death triad: `reward_std = grad_norm = kl = 0` every step.**
#    This is intra-group reward collapse. Loss can look "small but varying"
#    (0.001-0.003) and still be dead — that small loss is coming purely
#    from β×KL, not from advantage. When all three are zero, nothing is
#    moving and you are burning compute.
#
# 3. **Healthy signal**: `reward_std > 0` AND (`kl > 0` OR `grad_norm > 0`).
#    Loss values like `0.001277 → 0.000829 → 0.000995 → 0.001018` (non-zero
#    and wiggling) are healthy. Absolute magnitude is small because GRPO
#    advantage is group-normalized by construction.
#
# 4. **Manual rollout sanity check before you kick off.** Run
#    `model.generate(..., num_return_sequences=4, temperature=1.3,
#    do_sample=True)` on one prompt. Look at all 4 outputs. If they're
#    all near-identical, K=8 won't save you — SFT has collapsed the
#    distribution too hard. Either raise temperature, raise K, or lower β.
#
# 5. **Watchdog fires in 5 steps**: see the `RewardStdWatchdog` callback
#    above. If training aborts early, keep the SFT checkpoint and write
#    the run up as a negative result.
