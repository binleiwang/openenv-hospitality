"""
Lambda H100 adaptation of train_qwen7b_sft.py for Phase 4 / Phase 5.

USAGE (in Lambda JupyterLab, open this file and run cell-by-cell, OR run as script):

  python train_lambda.py

Differences from the Colab version:
  - No Google Drive; uses /home/ubuntu/hospitality for all paths
  - Parametric MODEL_NAME at top (switch 7B ↔ 14B ↔ 32B)
  - Explicit dtype=torch.bfloat16 (H100 is bf16-native)
  - Data loaded from sft_data/hospitality_sft_50.jsonl (threshold 0.5, 50 traj)
  - Eval loop SAVES FULL TRACES (was missing in Colab version; cost us the
    inability to do case-study diagnosis on today's eval regression)
  - HF Hub push repo name parametric based on MODEL_NAME

Run-order plan on Lambda:
  1. Run Phase 4 (MODEL_NAME = Qwen2.5-7B-Instruct), push to Hub as -sft-v2
  2. Kernel → Restart (frees 7B GPU memory)
  3. Change MODEL_NAME to Qwen2.5-14B-Instruct, re-run everything
  4. Push to Hub as qwen2.5-14b-hospitality-sft
  5. Terminate instance
"""

# %% [markdown]
# # Qwen SFT on Lambda H100 — Phase 4/5 Production Run
#
# Upstream of this notebook: baseline_eval.py must have already produced
# eval_results/baseline_claude-sonnet-4-5_20260421_002809.json (4.7MB with
# transcripts). Then build_sft_dataset.py --min-reward 0.5 was run to
# produce sft_data/hospitality_sft_50.jsonl (50 trajectories).

# %% [markdown]
# ## 1. Config — EDIT THIS CELL BETWEEN PHASE 4 AND PHASE 5

# %%
# ──────── CONFIG ────────
# 14B experiment Stage 2 — run AFTER eval_base_14b_stratified.py confirms
# that 14B base materially beats 7B base (Δ ≥ +0.05). If 14B base also ≈
# +0.10, skip this file entirely and write up the null result.
#
# 7B baseline grid (all on stratified eval):
#   7B base (no SFT)         +0.101
#   7B v1 (3ep, 37 traj)     +0.101  ← IDENTICAL to base per-task
#   7B v2 (3ep, 50 traj)     +0.133
#   7B v3 (6ep, 40 traj)     +0.104
#   Claude Sonnet 4.5        +0.314
#
# Hypothesis for 14B: if base 14B ≥ +0.20, SFT on top can push further.
# If base 14B ≈ +0.10, SFT won't help (same pattern as 7B).
MODEL_NAME = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"

# Hub repo: new name for the 14B lineage
HUB_REPO = "binleiwang/qwen2.5-14b-hospitality-sft"
# HUB_REPO = "binleiwang/qwen2.5-7b-hospitality-sft-v3p1"  # 7B v3.1 (never ran)
# HUB_REPO = "binleiwang/qwen2.5-7b-hospitality-sft-v3"    # 7B overfit
# HUB_REPO = "binleiwang/qwen2.5-7b-hospitality-sft-v2"    # 7B undertrained
# HUB_REPO = "binleiwang/qwen2.5-7b-hospitality-sft"       # 7B v1

# SFT data — SAME stratified pool as 7B v3. Only the base model changes.
# This is the cleanest A/B: isolate the effect of model capacity.
SFT_PATH = "/home/ubuntu/openenv-hospitality/sft_data/hospitality_sft_strat.jsonl"
EVAL_IDS_PATH = "/home/ubuntu/openenv-hospitality/sft_data/stratified_eval_ids.json"

# Local artifact paths (Lambda ephemeral disk — must push to Hub before Terminate)
LOCAL_BASE = "/home/ubuntu/hospitality"
OUTPUT_DIR = f"{LOCAL_BASE}/adapter_14b"
CHECKPOINT_DIR = f"{LOCAL_BASE}/checkpoints_14b"
EVAL_OUT = f"{LOCAL_BASE}/eval/eval_heldout_14b.json"

# Training config — conservative "v1 recipe" applied to 14B:
# - 3 epochs × 40 traj / 8 accum = 15 optimizer steps (same as v1/v3.1 would be)
# - LR 2e-4 (not 3e-4): larger model → smaller relative per-step updates safer
# - warmup_ratio 0.05: give the larger param set a brief warmup
# Why conservative? We want to isolate "does bigger base + same recipe help?"
# not conflate with LR / epoch changes. If this works, tune later; if not,
# we've kept the comparison interpretable.
MAX_SEQ_LENGTH = 8192
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
SEED = 42

# %%
import os
os.makedirs(LOCAL_BASE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(f"{LOCAL_BASE}/eval", exist_ok=True)

# CUDA memory fragmentation guard (learned from OOM saga — see §负零点负一)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Lambda's stock image ships TF + Keras 3 + JAX + Flax pre-installed globally.
# transformers.modeling_tf_utils eagerly probes Keras and refuses Keras 3;
# the import chain then dies with "please install tf-keras". We don't use any
# of these frameworks — tell transformers to skip the TF/JAX/Flax code paths
# entirely. Must be set BEFORE `import transformers`.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_JAX"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# Silence TF's CUDA registration warnings (cuFFT/cuDNN/cuBLAS registered twice)
# when TF loads alongside torch. TF still loads (we can't prevent it — system
# install), but its logs are hidden.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print(f"MODEL_NAME: {MODEL_NAME}")
print(f"HUB_REPO:   {HUB_REPO}")
print(f"SFT_PATH:   {SFT_PATH}")
print(f"LOCAL_BASE: {LOCAL_BASE}")

# %% [markdown]
# ## 2. Install dependencies
#
# Lambda's stock PyTorch image has CUDA + base Torch. We layer on:
# - Unsloth (4-bit LoRA training, bf16-native on H100)
# - trl 0.15.2 (SFTTrainer + GRPOTrainer compatible with Unsloth pin)
# - Transformers 4.51+ (Qwen2.5 support)
# - The hospitality_env package itself (installed non-editable because
#   Lambda's stock setuptools < PEP 660 era; editable fails with
#   "build backend is missing the 'build_editable' hook").
#
# Only run this cell ONCE per instance (pip is idempotent anyway).

# %%
# ─────────────────────────────────────────────────────────────
# Full audit of pin rationale (lessons from 3 prior install failures):
#
#   torch==2.10.0       Lambda stock image has 2.10; keep it. Don't let
#                       torchvision>=0.25 drift torch to 2.11 (unsloth caps
#                       torch<2.11.0).
#   torchvision==0.25.0 Required by torch 2.10. Stock image has 0.22.
#   numpy==1.26.4       Ubuntu 22.04 system numpy is 1.21.x, lacks modern
#                       __class_getitem__ support. transformers 4.51 needs 1.26+.
#   Pillow==10.4.0      Ubuntu 22.04 system PIL is 7.x, lacks Image.Resampling
#                       (added in PIL 9.0). transformers.image_utils crashes.
#   transformers==4.51.3  Range >=4.51.3,<=5.5.0 pulls 4.5x which has new
#                       nested np.ndarray[np.ndarray[...]] annotations that
#                       don't eval on older numpy. 4.51.3 is Colab-tested.
#   trl==0.18.2         Minimum that satisfies unsloth 2026.4.6; API is
#                       processing_class-based, matches our notebook.
#   accelerate==1.6.0   Colab-tested. 1.13 (auto-installed by unsloth) works
#                       but not guaranteed stable with trl 0.18.2.
#   bitsandbytes==0.46.1  Latest stable as of 2026Q2.
#   pydantic==2.10.6    Colab-tested. Openenv-core 0.2.3 requires 2.10.x.
#   datasets==3.6.0     Colab-tested.
#   openenv-core==0.2.3 Colab-tested.
#   unsloth==2026.4.6   Latest as of now. Needs trl>=0.18.2 and torch<2.11.
# ─────────────────────────────────────────────────────────────

# Upgrade build tooling first (editable installs, PEP 660)
!pip install -q --upgrade pip setuptools wheel

# Nuke legacy versions that survived earlier install attempts
!pip uninstall -y -q trl transformers accelerate || true

# Handle Lambda's pre-installed TF/Keras conflict.
#
# Background: Lambda stock image has tensorflow + Keras 3 installed SYSTEM-WIDE
# at /usr/lib/python3/dist-packages/tensorflow (installed via Debian apt).
# We CANNOT uninstall these without sudo. So even with USE_TF=0, the import
# chain `from transformers import Trainer` eagerly hits
# transformers/activations_tf.py, which does `import tf_keras OR import keras`.
# With tf_keras missing, it falls back to Keras 3 which transformers rejects.
#
# Fix: install tf-keras (the backwards-compat shim transformers looks for first).
# This lets activations_tf succeed on the `import tf_keras` line and skip the
# Keras 3 code path entirely. TF still loads (eating ~500MB VRAM on H100 —
# harmless on 80GB) but the import chain completes.
!pip install -q tf-keras

# --- Core numerics stack (pinned exactly, installed first) ---
!pip install -q "numpy==1.26.4" "Pillow==10.4.0"
!pip install -q "torch==2.10.0" "torchvision==0.25.0"

# --- Unsloth (latest PyPI, not git+https which has #egg fragment bug) ---
!pip install -q unsloth==2026.4.6

# --- Training stack: EXACT pins, no ranges ---
!pip install -q "transformers==4.51.3" "trl==0.18.2" "accelerate==1.6.0" "bitsandbytes==0.46.1"
!pip install -q "pydantic==2.10.6" "pydantic-core==2.27.2"
!pip install -q "datasets==3.6.0"

# --- Env server + client runtime ---
!pip install -q "openenv-core==0.2.3" fastapi "uvicorn[standard]" httpx nest_asyncio

# --- Template / hub / tokenizers baseline (Lambda stock is too old) ---
# Lambda image ships jinja2 3.0.3; transformers.apply_chat_template needs >=3.1.0
# (raises ImportError with the exact version string otherwise).
# Pin conservatively — these are small, stable libs.
!pip install -q "jinja2>=3.1.4" "markupsafe>=2.1.5" "huggingface_hub>=0.24.0" \
                 "tokenizers>=0.21.0" "typing_extensions>=4.10.0" "safetensors>=0.4.3"

# --- Install hospitality_env itself (non-editable, avoids PEP 660 issue) ---
!pip install --no-deps -q /home/ubuntu/openenv-hospitality/hospitality_env/

# --- Post-install verification (fail LOUD if any pin drifted) ---
import importlib, sys

# If kernel wasn't restarted after an earlier failed install, stale modules
# may still be loaded. Force-reload the critical ones.
for _mod in ("transformers", "trl", "accelerate", "unsloth", "numpy", "PIL"):
    if _mod in sys.modules:
        try:
            importlib.reload(sys.modules[_mod])
        except Exception:
            pass  # some fail reload; kernel restart is the canonical fix

import unsloth, torch, transformers, trl, accelerate, numpy, PIL
import hospitality_env
from hospitality_env import HospitalityAction

# Strict assertions — if ANY of these fail, we're building on sand
_versions = {
    "torch":        torch.__version__,
    "torchvision":  __import__("torchvision").__version__,
    "numpy":        numpy.__version__,
    "Pillow":       PIL.__version__,
    "transformers": transformers.__version__,
    "trl":          trl.__version__,
    "accelerate":   accelerate.__version__,
    "unsloth":      unsloth.__version__,
}
print("Installed versions:")
for k, v in _versions.items():
    print(f"  {k:14s} {v}")

# Hard expectations (prints, doesn't crash — kernel restart may be needed)
_expect = {
    "torch":        "2.10.",
    "transformers": "4.51.3",
    "trl":          "0.18.2",
    "unsloth":      "2026.4.6",
    "numpy":        "1.26.",
    "Pillow":       "10.",
}
_mismatches = [k for k, prefix in _expect.items() if not _versions[k].startswith(prefix)]
if _mismatches:
    print(f"\n⚠️  Version mismatch: {_mismatches}")
    print("   → Kernel → Restart Kernel, then re-run this cell.")
else:
    print(f"\n✓ All pins correct. CUDA: {torch.cuda.is_available()}, "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

# %% [markdown]
# ## 2b. HuggingFace login + rebuild STRATIFIED SFT dataset
#
# Phase 5 (v3) uses STRATIFIED data: per-category top-K from Claude baseline.
#
# Why stratified?  Phase 4 (threshold-0.5 global filter) produced +0.133 mean
# reward — WORSE than v1 (+0.338). Diagnosis: the threshold filter drops
# Claude's weak categories entirely (host_seating had 0 tasks above 0.5), so
# the SFT model never sees those decision patterns and fails on them at eval.
#
# Fix: per-category top-K (k = max(2, floor(n*0.5))), no global threshold.
# Even Claude's worst category (host_seating, max reward +0.50) contributes
# 2 demonstrations. Pool size: ~40 trajectories, but all 11 categories covered.
#
# Also builds stratified eval set: 20% holdout per category, ~20 tasks total.
# Claude on this NEW eval = +0.314 (NOT the +0.604 we were comparing against).
# That earlier number was inflated by ID-order bias.

# %%
import sys, subprocess
from huggingface_hub import whoami
try:
    _who = whoami()
    print(f"HF logged in as: {_who['name']}")
except Exception:
    print("Not logged in. Run: huggingface-cli login   (in a terminal)")
    raise

# %%
# Build stratified data — idempotent, safe to re-run
_build = subprocess.run(
    [sys.executable, "build_stratified_data.py"],
    cwd="/home/ubuntu/openenv-hospitality",
    capture_output=True, text=True,
)
print(_build.stdout)
if _build.returncode != 0:
    print(_build.stderr); raise RuntimeError("build_stratified_data failed")

# Verify files
import json as _json
with open(SFT_PATH) as f:
    _sft_n = sum(1 for _ in f)
_eval_ids = _json.load(open(EVAL_IDS_PATH))
print(f"\nStratified SFT pool: {_sft_n} trajectories")
print(f"Stratified eval set: {len(_eval_ids)} tasks")
assert 30 <= _sft_n <= 60, f"Unexpected SFT pool size {_sft_n}"
assert 15 <= len(_eval_ids) <= 30, f"Unexpected eval size {len(_eval_ids)}"

# Print category coverage for sanity
_rows = [_json.loads(line) for line in open(SFT_PATH)]
from collections import Counter
_cat_dist = Counter(r["category"] for r in _rows)
print(f"\nSFT pool category coverage ({len(_cat_dist)} categories):")
for cat, n in sorted(_cat_dist.items()):
    print(f"  {cat:24s} {n}")

# %%
# Stubs for vllm (trl 0.15.2 conditionally imports it; we don't actually use it)
import os, sys, types, importlib.machinery, pathlib
import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

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
# Install the stub so `import vllm` inside trl doesn't blow up
if "vllm" not in sys.modules:
    try:
        import vllm  # already installed somehow
    except ImportError:
        import subprocess as _sp
        _sp.check_call([sys.executable, "-m", "pip", "install", "-q", STUB_DIR])

import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
# ## 3. Launch the env server

# %%
import subprocess, time, httpx, sys

# Use sys.executable so the subprocess inherits THIS kernel's Python + site-packages.
# A bare "python" on Lambda resolves to /usr/bin/python3 (system), which does NOT
# have our pip-installed hospitality_env / uvicorn / fastapi — the server would
# silently fail to import. sys.executable is the canonical fix.
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "hospitality_env.server.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd="/home/ubuntu/openenv-hospitality",
)
for _ in range(30):
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
        if r.status_code == 200:
            print("Server up:", r.json()); break
    except Exception:
        time.sleep(1)
else:
    # Drain whatever the server emitted before giving up
    try:
        out_tail = server_proc.stdout.read(4000).decode(errors="ignore") if server_proc.stdout else ""
        err_tail = server_proc.stderr.read(4000).decode(errors="ignore") if server_proc.stderr else ""
    except Exception:
        out_tail, err_tail = "", ""
    raise RuntimeError(
        f"Server failed to come up in 30s.\n"
        f"stdout tail:\n{out_tail[-2000:]}\n"
        f"stderr tail:\n{err_tail[-2000:]}"
    )

# %% [markdown]
# ## 4. Load model — H100 uses bf16-native

# %%
from unsloth import FastLanguageModel
from transformers import set_seed
import torch, random, numpy as np

set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# EXPLICIT bfloat16 — do NOT rely on dtype=None on H100, it occasionally
# selects fp16 and causes slow + numerically unstable training.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=torch.bfloat16,
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
# ## 5. SFT — 50 trajectories from threshold 0.5 filter

# %%
import json
from datasets import Dataset

sft_rows = [json.loads(line) for line in open(SFT_PATH)]
print(f"SFT examples: {len(sft_rows)} (expected 50)")
print(f"Mean turns:   {sum(r['turns'] for r in sft_rows) / len(sft_rows):.1f}")
print(f"Mean reward:  {sum(r['reward'] for r in sft_rows) / len(sft_rows):.2f}")

def _format(row):
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

sft_ds = Dataset.from_list(sft_rows).map(_format, remove_columns=["messages"])

# Sanity check — prompt lengths
_lens = [len(tokenizer(r["text"]).input_ids) for r in sft_ds]
print(f"Prompt token lengths: min={min(_lens)}, "
      f"mean={sum(_lens)/len(_lens):.0f}, max={max(_lens)}")
# Note: longer examples get silently truncated at max_seq_length=8192.
# This is intentional — Phase 3's reference 0.659 result was under this
# truncation. See dev notes §负零点负一 rule C.

# %%
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,   # trl 0.18+ defaults to 1024 if unset → truncates aggressively
    packing=False,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=sft_ds,
)
sft_trainer.train()

# Expected training loss:
#   7B:   step 1 ~2.5, step final ~0.25-0.35 (3 epoch on 50 traj)
#   14B:  step 1 ~2.0, step final ~0.15-0.25
# If final loss > 0.5, SFT undertrained. If < 0.15, may be overfitting
# (but eval is the real signal — see §负零点零续 rule H).

# %% [markdown]
# ## 6. Agent wrapper (SYSTEM_PROMPT_V2 + anti-loop + rollout_v3)

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
# ## 7. Held-out eval — WITH TRACE SAVING THIS TIME

# %%
import asyncio, torch, statistics as st
from pathlib import Path
from hospitality_env.client import HospitalityEnv

TASKS_PATH = Path("/home/ubuntu/openenv-hospitality/hospitality_env/server/data/tasks.json")
ALL_TASKS = json.load(open(TASKS_PATH))

# STRATIFIED eval set from v3 (20% holdout per category, ~20 tasks covering
# all 10 multi-task categories). Replaces the old ID-ordered tasks[100:115]
# slice which was biased toward server_misc + food_issue.
EVAL_IDS = json.load(open(EVAL_IDS_PATH))
TID_TO_CAT = {t["id"]: t["description"]["category"] for t in ALL_TASKS}

# Print eval set distribution so we can SEE it's stratified
from collections import Counter as _Counter
print(f"Stratified eval set: {len(EVAL_IDS)} tasks, covering "
      f"{len(set(TID_TO_CAT[tid] for tid in EVAL_IDS))} categories")
for cat, n in sorted(_Counter(TID_TO_CAT[tid] for tid in EVAL_IDS).items()):
    print(f"  {cat:24s} {n}")

FastLanguageModel.for_inference(model)

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
    """Multi-turn rollout with anti-loop nudge AND full trace capture."""
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    total_reward, turns, tool_calls = 0.0, 0, 0
    no_tool_streak = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2}]
    trace = []   # ← NEW: per-turn record for diagnosis
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

            # NEW: capture per-turn state for post-hoc diagnosis
            trace.append({
                "turn": turn,
                "customer_msg_preview": (obs.customer_message or "")[:200],
                "raw_response": raw[:500],
                "tool_name": action.tool_name,
                "tool_args": action.tool_args,
                "message": (action.message or "")[:300],
                "step_reward": step.reward,
                "done": step.done,
            })

            if verbose:
                print(f"  turn {turn}: tool={action.tool_name!r} r={step.reward:+.2f}")
            if step.done:
                break
            obs = step.observation
        return {"task_id": tid, "reward": total_reward, "turns": turns,
                "tool_calls": tool_calls, "trace": trace}
    except Exception as e:
        return {"task_id": tid, "reward": None, "turns": turns,
                "tool_calls": tool_calls, "error": str(e)[:200],
                "trace": trace}
    finally:
        await client.close()


# %%
# Run eval
results = []
for tid in EVAL_IDS:
    r = asyncio.run(rollout_v3(tid))
    r["category"] = TID_TO_CAT.get(tid, "unknown")
    results.append(r)
    if r.get("error"):
        print(f"{tid:50s} [{r['category']:20s}] ERROR {r['error']}")
    else:
        print(f"{tid:50s} [{r['category']:20s}] "
              f"r={r['reward']:+.3f} turns={r['turns']} tools={r['tool_calls']}")

# Summary
valid = [r for r in results if r["reward"] is not None]
rewards = [r["reward"] for r in valid]
mean_reward = st.mean(rewards)
print(f"\n=== Held-out eval ({MODEL_NAME.split('/')[-1]}) on STRATIFIED set ===")
print(f"Mean:     {mean_reward:+.3f}")
print(f"Median:   {st.median(rewards):+.3f}")
print(f"Min/Max:  {min(rewards):+.3f} / {max(rewards):+.3f}")
print(f"Mean turns: {st.mean([r['turns'] for r in valid]):.2f}")
print(f"\n=== Context: all on SAME stratified 20-task eval ===")
print(f"Claude Sonnet 4.5:            +0.314  ← target")
print(f"7B base (no SFT):             +0.101")
print(f"7B v1 (SFT, 37 traj):         +0.101  ← identical to base, per-task")
print(f"7B v2 (SFT, 50 traj):         +0.133")
print(f"7B v3 (SFT, 40 traj, strat):  +0.104")
print(f"14B SFT (THIS RUN):           {mean_reward:+.3f}")
if MODEL_NAME.startswith("unsloth/Qwen2.5-14B"):
    _delta_vs_7b_base = mean_reward - 0.101
    print(f"\n14B SFT vs 7B base Δ = {_delta_vs_7b_base:+.3f}")
    if _delta_vs_7b_base >= 0.15:
        print("  → 14B + SFT cracks the reasoning barrier. Ship this.")
    elif _delta_vs_7b_base >= 0.05:
        print("  → 14B + SFT helps some. Worth shipping as best checkpoint.")
    else:
        print("  → 14B + SFT does not meaningfully help. Null result story.")

# Per-category breakdown — THIS is the diagnostic goldmine
from collections import defaultdict as _dd
by_cat = _dd(list)
for r in valid:
    by_cat[r["category"]].append(r["reward"])

# Also pull Claude's per-category mean from the baseline for comparison
_baseline_path = "/home/ubuntu/openenv-hospitality/eval_results/baseline_claude-sonnet-4-5_20260421_002809.json"
_claude_records = {r["task_id"]: r["reward"]
                   for r in json.load(open(_baseline_path))["records"]}
claude_by_cat = _dd(list)
for tid in EVAL_IDS:
    claude_by_cat[TID_TO_CAT[tid]].append(_claude_records[tid])

print(f"\n=== Per-category breakdown ===")
print(f"{'category':24s} {'n':>3s}  {'ours':>7s}  {'Claude':>7s}  {'Δ':>7s}  {'verdict':>10s}")
print("-" * 80)
cat_rows = []
for cat in sorted(by_cat.keys()):
    ours = st.mean(by_cat[cat])
    cla = st.mean(claude_by_cat[cat]) if cat in claude_by_cat else float("nan")
    diff = ours - cla
    verdict = ("BEAT" if diff > 0.05 else
               "MATCH" if abs(diff) <= 0.05 else
               "LOSE")
    cat_rows.append((cat, len(by_cat[cat]), ours, cla, diff, verdict))
    print(f"{cat:24s} {len(by_cat[cat]):>3d}  "
          f"{ours:>+7.3f}  {cla:>+7.3f}  {diff:>+7.3f}  {verdict:>10s}")

# %%
# Save eval results WITH TRACES + per-category breakdown
with open(EVAL_OUT, "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "hub_repo": HUB_REPO,
        "sft_path": SFT_PATH,
        "eval_set": "stratified_v3",
        "n_eval": len(EVAL_IDS),
        "claude_on_same_eval": 0.314,
        "results": results,
        "per_category": [
            {"category": c, "n": n, "ours": o, "claude": cl, "delta": d, "verdict": v}
            for c, n, o, cl, d, v in cat_rows
        ],
        "summary": {
            "mean": mean_reward,
            "median": st.median(rewards),
            "min": min(rewards),
            "max": max(rewards),
            "mean_turns": st.mean([r['turns'] for r in valid]),
            "n": len(valid),
            "seed": SEED,
        },
    }, f, indent=2, default=str)
print(f"\nWrote {EVAL_OUT}")

# %% [markdown]
# ## 8. Save + push to HF Hub
#
# IMPORTANT: Lambda instance disk is ephemeral — you MUST push to Hub before
# Terminate. Otherwise the adapter is lost.

# %%
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved adapter + tokenizer to {OUTPUT_DIR}")

# %%
from huggingface_hub import HfApi, login
# login()  # uncomment if not already logged in via CLI

commit_msg = (
    f"SFT {NUM_EPOCHS}ep LR={LEARNING_RATE} | {sum(1 for _ in open(SFT_PATH))} traj "
    f"(stratified per-category) | stratified eval {mean_reward:+.3f} "
    f"vs Claude {0.314:+.3f} on same eval"
)
HfApi().upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=HUB_REPO,
    repo_type="model",
    commit_message=commit_msg,
)
print(f"Pushed to: https://huggingface.co/{HUB_REPO}")
print(f"Commit: {commit_msg}")

# Also upload the eval record so the per-task breakdown lives next to the model
HfApi().upload_file(
    path_or_fileobj=EVAL_OUT,
    path_in_repo="eval_heldout.json",
    repo_id=HUB_REPO,
    repo_type="model",
    commit_message=f"eval record: mean {mean_reward:.3f}, with per-turn traces",
)
print("Eval record uploaded.")

# %% [markdown]
# ## 9. RFT — Rejection-Sampled Fine-Tuning (Stage B, primary)
#
# **Why RFT instead of real GRPO**:
# - trl 0.15.2 `GRPOTrainer` is single-turn native. Multi-turn tool-use GRPO
#   is research territory and requires custom rollout integration.
# - Our two previous GRPO attempts (Phase 1: 1.5B, Phase 2: 7B+1ep) both
#   collapsed with `reward_std=0`. Even with a stronger Phase 4 foundation,
#   GRPO on this long-horizon env is risky.
# - RFT is "GRPO's stable cousin": sample K rollouts from current model,
#   keep only the high-reward ones, SFT another epoch on the kept set.
#   This is the recipe used in the Llama-2 paper (§5 Reward Modeling + RFT).
# - RFT cannot collapse to std=0 because it's fundamentally SFT.
#
# **Gate**: only run RFT if Phase 4 eval >= 0.40, else ship v1 (0.338).
#
# **Budget**: ~50 min for 100 tasks × K=3 rollouts + ~20 min for SFT + eval.

# %%
# --- Stage-B gate ---
_RFT_MIN = 0.40
if mean_reward < _RFT_MIN:
    print(f"Phase 4 eval {mean_reward:.3f} < {_RFT_MIN} — SKIP RFT, ship Phase 4.")
    print("(If you want to force RFT anyway, comment out the raise below.)")
    raise SystemExit(f"Gated out of RFT at {mean_reward:.3f}")
print(f"Phase 4 eval {mean_reward:.3f} >= {_RFT_MIN} — proceeding to RFT.")

# %%
# Sampling-mode generator (diverse rollouts for RFT pool collection)
def generate_text_sample(messages, max_new_tokens=384, temperature=0.7, top_p=0.9):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LENGTH - max_new_tokens,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


async def rollout_sample(tid, max_turns=8, temperature=0.7):
    """Multi-turn rollout with sampling. Captures full message list for RFT."""
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    total_reward, turns = 0.0, 0
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
            raw = generate_text_sample(messages, temperature=temperature)
            messages.append({"role": "assistant", "content": raw})

            action = parse_action(raw)
            if action.tool_name:
                no_tool_streak = 0
            else:
                no_tool_streak += 1

            step = await client.step(action)
            total_reward += step.reward or 0.0
            turns += 1
            if step.done:
                break
            obs = step.observation
        return {"task_id": tid, "reward": total_reward, "turns": turns,
                "messages": messages}
    except Exception as e:
        return {"task_id": tid, "reward": None, "turns": turns,
                "messages": messages, "error": str(e)[:200]}
    finally:
        await client.close()

# %%
# --- Collect RFT pool: K=3 rollouts per training task ---
TRAIN_IDS = [t["id"] for t in ALL_TASKS[:100]]
RFT_K = 3
RFT_KEEP_THRESHOLD = 0.5   # keep rollouts with reward > this

FastLanguageModel.for_inference(model)

rft_pool = []
rft_stats = []   # per-task stats for diagnosis
for i, tid in enumerate(TRAIN_IDS):
    task_rollouts = []
    for _ in range(RFT_K):
        r = asyncio.run(rollout_sample(tid, temperature=0.7))
        task_rollouts.append(r)
    valid = [r for r in task_rollouts if r.get("reward") is not None]
    kept = [r for r in valid if r["reward"] > RFT_KEEP_THRESHOLD]
    rft_pool.extend(kept)
    best = max((r["reward"] for r in valid), default=None)
    rft_stats.append({
        "task_id": tid, "k_valid": len(valid),
        "k_kept": len(kept), "best_r": best,
    })
    if (i + 1) % 10 == 0 or i == len(TRAIN_IDS) - 1:
        print(f"  [{i+1}/{len(TRAIN_IDS)}] pool={len(rft_pool)} "
              f"(avg kept/task={len(rft_pool)/(i+1):.2f})")

print(f"\n=== RFT pool collection done ===")
print(f"Total rollouts: {len(TRAIN_IDS) * RFT_K}")
print(f"Kept (reward > {RFT_KEEP_THRESHOLD}): {len(rft_pool)}")
print(f"Task coverage: {sum(1 for s in rft_stats if s['k_kept'] > 0)}/{len(TRAIN_IDS)}")

# Save pool for reproducibility
RFT_POOL_PATH = f"{LOCAL_BASE}/rft_pool.jsonl"
with open(RFT_POOL_PATH, "w") as f:
    for r in rft_pool:
        f.write(json.dumps({
            "task_id": r["task_id"], "reward": r["reward"],
            "turns": r["turns"], "messages": r["messages"],
        }) + "\n")
print(f"Saved pool to {RFT_POOL_PATH}")

# %%
# --- Build combined RFT training dataset (original 50 + self-rollouts) ---
# Note: original sft_rows already has messages from Claude.
# We combine Claude demos + our model's own high-reward rollouts.
combined_rows = list(sft_rows)   # Phase 4 data (50 Claude traj, threshold 0.5)
for r in rft_pool:
    combined_rows.append({
        "task_id": r["task_id"],
        "reward": r["reward"],
        "turns": r["turns"],
        "messages": r["messages"],
    })

print(f"RFT combined dataset: {len(combined_rows)} "
      f"(Claude {len(sft_rows)} + self-rollout {len(rft_pool)})")

rft_ds = Dataset.from_list(combined_rows).map(_format, remove_columns=["messages"])

# %%
# --- RFT training: 1 epoch at lower LR (avoid overfitting Phase 4 into RFT) ---
# State transition: rollout collection ran in inference mode (for_inference flips
# requires_grad off on LoRA adapters). We must explicitly flip back, otherwise
# SFTTrainer gets a model with no trainable params → silently no-op train.
FastLanguageModel.for_training(model)
model.train()
# Paranoia check: at least one LoRA adapter parameter must be trainable
_n_trainable = sum(p.requires_grad for p in model.parameters())
assert _n_trainable > 0, (
    "No trainable parameters after for_training(). "
    "LoRA adapters may not have been re-enabled. Restart kernel."
)
print(f"RFT ready: {_n_trainable} trainable tensors, model.training={model.training}")

rft_config = SFTConfig(
    output_dir=f"{LOCAL_BASE}/checkpoints_rft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,           # half of Phase 4's 2e-4
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

rft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=rft_config,
    train_dataset=rft_ds,
)
rft_trainer.train()
# Expected loss trajectory: starts ~0.3-0.5 (model already partly fit),
# ends ~0.15-0.25 after 1 epoch.

# %%
# --- Eval post-RFT ---
FastLanguageModel.for_inference(model)

rft_results = []
for tid in EVAL_IDS:
    r = asyncio.run(rollout_v3(tid))
    rft_results.append(r)
    if r.get("error"):
        print(f"{tid}: ERROR {r['error']}")
    else:
        print(f"{tid}: r={r['reward']:+.3f} turns={r['turns']} tools={r['tool_calls']}")

rft_valid = [r for r in rft_results if r["reward"] is not None]
rft_rewards = [r["reward"] for r in rft_valid]
rft_mean = st.mean(rft_rewards)
rft_median = st.median(rft_rewards)

print(f"\n=== RFT held-out eval ===")
print(f"Mean:     {rft_mean:+.3f}   (Phase 4 was {mean_reward:+.3f})")
print(f"Median:   {rft_median:+.3f}")
print(f"Min/Max:  {min(rft_rewards):+.3f} / {max(rft_rewards):+.3f}")
print(f"Delta:    {rft_mean - mean_reward:+.3f}")
print(f"Claude baseline: +0.604")

# Save RFT eval with traces
RFT_EVAL_OUT = f"{LOCAL_BASE}/eval/eval_rft.json"
with open(RFT_EVAL_OUT, "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "stage": "RFT",
        "phase4_mean": mean_reward,
        "rft_mean": rft_mean,
        "rft_pool_size": len(rft_pool),
        "results": rft_results,
        "pool_stats": rft_stats,
        "summary": {
            "mean": rft_mean, "median": rft_median,
            "min": min(rft_rewards), "max": max(rft_rewards),
            "mean_turns": st.mean([r['turns'] for r in rft_valid]),
            "n": len(rft_valid), "seed": SEED,
        },
    }, f, indent=2, default=str)
print(f"Wrote {RFT_EVAL_OUT}")

# %%
# --- Push RFT adapter to Hub ---
RFT_HUB_REPO = HUB_REPO + "-rft"
RFT_OUTPUT_DIR = f"{LOCAL_BASE}/adapter_rft"

model.save_pretrained(RFT_OUTPUT_DIR)
tokenizer.save_pretrained(RFT_OUTPUT_DIR)

HfApi().upload_folder(
    folder_path=RFT_OUTPUT_DIR,
    repo_id=RFT_HUB_REPO,
    repo_type="model",
    commit_message=(
        f"RFT 1ep LR=1e-4 | {len(rft_pool)} self-rollout traj "
        f"(reward > {RFT_KEEP_THRESHOLD}) + {len(sft_rows)} Claude | "
        f"eval {rft_mean:.3f}"
    ),
)
HfApi().upload_file(
    path_or_fileobj=RFT_EVAL_OUT,
    path_in_repo="eval_heldout.json",
    repo_id=RFT_HUB_REPO,
    repo_type="model",
    commit_message=f"RFT eval: mean {rft_mean:.3f}, with per-turn traces",
)
print(f"Pushed RFT to https://huggingface.co/{RFT_HUB_REPO}")

# %% [markdown]
# ## 10. (Experimental) GRPO smoke test — run ONLY if you have time
#
# **Status**: research-territory. trl 0.15.2 GRPOTrainer treats the problem
# as single-turn (prompt → one completion → scalar reward). To apply it
# to our multi-turn env we define the reward as "execute this generated
# first action, then continue rollout greedily with the current model,
# return total trajectory reward".
#
# **Known risks**:
# 1. `reward_std = 0` collapse (our 2 previous failures)
# 2. Greedy continuation uses the SAME model that's being trained, so
#    the reward signal drifts mid-training
# 3. trl may not forward custom dataset columns to `reward_funcs`
# 4. Each GRPO step calls the env, which is slow
#
# **Protocol**:
# 1. Run 10-step smoke test below
# 2. After each step, print `reward_std` from logs
# 3. If std=0 across 5+ consecutive steps → STOP, GRPO won't help
# 4. If std>0 → consider a longer run (100-200 steps), but set a 2-hour
#    wall-clock timeout
#
# **If you're tight on time, SKIP THIS SECTION. Ship the RFT model.**

# %%
# Escape hatch: comment this out to actually run GRPO smoke test
RUN_GRPO_SMOKE = False   # flip to True only if you have >1 hour buffer
if not RUN_GRPO_SMOKE:
    print("GRPO smoke test disabled (set RUN_GRPO_SMOKE=True to enable).")
    print("Ship the RFT model from Section 9.")
    raise SystemExit("GRPO opt-in not set")

# %%
# --- Build GRPO prompts dataset ---
from trl import GRPOConfig, GRPOTrainer

async def _get_init_prompt(tid):
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    try:
        step = await client.reset(task_id=tid)
        obs = step.observation
        user_content = build_user_turn(obs_to_dict(obs), first_turn=True)
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": user_content},
        ]
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_str, "task_id": tid}
    finally:
        await client.close()

# Smoke test uses only 20 training tasks
SMOKE_IDS = TRAIN_IDS[:20]
grpo_rows = [asyncio.run(_get_init_prompt(tid)) for tid in SMOKE_IDS]
grpo_ds = Dataset.from_list(grpo_rows)

# %%
# --- Reward function: completion = generated first action ---
# For each (prompt, completion) pair, replay the environment starting
# with that first action, then continue rollout greedily, return total reward.

def grpo_reward_fn(prompts, completions, **kwargs):
    task_ids = kwargs.get("task_id", [None] * len(completions))
    rewards = []

    async def _rollout_one(tid, first_completion):
        client = HospitalityEnv(base_url="http://127.0.0.1:8000")
        try:
            step = await client.reset(task_id=tid)
            total = 0.0
            # Execute the generated first action
            action_0 = parse_action(first_completion)
            step = await client.step(action_0)
            total += step.reward or 0.0
            if step.done:
                return total
            # Greedy continuation for remaining turns
            obs = step.observation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": build_user_turn(
                    obs_to_dict(obs), first_turn=True)},
                {"role": "assistant", "content": first_completion},
            ]
            for _ in range(1, 8):
                messages.append({"role": "user", "content": build_user_turn(
                    obs_to_dict(obs), first_turn=False)})
                raw = generate_text(messages)
                messages.append({"role": "assistant", "content": raw})
                step = await client.step(parse_action(raw))
                total += step.reward or 0.0
                if step.done:
                    break
                obs = step.observation
            return total
        except Exception as e:
            return 0.0
        finally:
            await client.close()

    for tid, comp in zip(task_ids, completions):
        rewards.append(asyncio.run(_rollout_one(tid, comp)))
    return rewards

# %%
# --- GRPO smoke config: 10 steps, K=4 rollouts per prompt ---
FastLanguageModel.for_training(model)
model.train()
assert sum(p.requires_grad for p in model.parameters()) > 0, \
    "No trainable params before GRPO. Restart kernel."

grpo_config = GRPOConfig(
    output_dir=f"{LOCAL_BASE}/checkpoints_grpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,            # K rollouts per prompt
    max_prompt_length=4096,
    max_completion_length=384,
    num_train_epochs=1,
    max_steps=10,                 # smoke: 10 steps
    learning_rate=5e-6,           # conservative; RL is sensitive
    lr_scheduler_type="constant",
    logging_steps=1,
    bf16=True,
    report_to="none",
    temperature=0.7,
    seed=SEED,
    use_vllm=False,               # we have no vllm
    remove_unused_columns=False,  # keep task_id for reward fn
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=grpo_reward_fn,
    args=grpo_config,
    train_dataset=grpo_ds,
    processing_class=tokenizer,
)

# Run and WATCH the `reward_std` log line per step
# If std=0 for 5+ steps, INTERRUPT the kernel and ship the RFT model.
grpo_trainer.train()

# After 10 steps, check logs. Decision:
#   - `reward_std` > 0.1 average: GRPO is learning, consider full run
#   - `reward_std` < 0.05 average: collapse mode, ship RFT
print("GRPO smoke test done. Check the logged `reward_std` values above.")
print("If healthy, re-run with max_steps=200 + push to -sft-grpo Hub repo.")
print("If collapsed, ship RFT (Section 9 adapter).")

# %% [markdown]
# ## 11. AFTER ALL STAGES — Terminate Lambda instance
#
# From Lambda dashboard:
#   Instances → find running → **Terminate** (NOT Stop)
#
# Verify "Remaining service credit" decreased by ~$12-20 for this session.
#
# Expected Hub artifacts after a successful run:
#   - binleiwang/qwen2.5-7b-hospitality-sft       (v1, already pushed, safety net)
#   - binleiwang/qwen2.5-7b-hospitality-sft-v2    (Phase 4 SFT)
#   - binleiwang/qwen2.5-7b-hospitality-sft-v2-rft (RFT — main deliverable)
#   - binleiwang/qwen2.5-7b-hospitality-sft-v2-grpo (optional, only if smoke passed)
