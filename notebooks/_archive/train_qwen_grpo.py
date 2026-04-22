"""
Train Qwen2.5-1.5B on the Hospitality Env (a Hot Pot Restaurant Simulation) with GRPO.

Naming convention used throughout this project:
  - Formal name: Hospitality RL Environment: A Hot Pot Restaurant Simulation
  - Short name:  Hospitality Env
  - Code ID:     hospitality_env / HospitalityEnv

This file uses jupytext percent format (`# %%` cell markers). Open directly in
VS Code (Jupyter extension) or Colab, or convert to .ipynb with:

    pip install jupytext
    jupytext --to notebook train_qwen_grpo.py

Runtime target: Google Colab free tier (T4 GPU, 16 GB VRAM).

Pipeline:
    1. Install deps (unsloth, trl, openenv-core, hospitality_env)
    2. Launch hospitality_env FastAPI server in background
    3. Load Qwen2.5-1.5B with Unsloth 4-bit + LoRA adapters
    4. Define rollout: single-episode env interaction with current policy
    5. Define reward function for GRPOTrainer
    6. Train (small step budget for Colab free tier)
    7. Save LoRA adapter + push to Hugging Face Hub
    8. Run quick eval on held-out tasks
"""

# %% [markdown]
# # Training Qwen2.5-1.5B on the Hospitality Env (GRPO)
#
# An end-to-end fine-tuning recipe for the
# **Hospitality RL Environment: A Hot Pot Restaurant Simulation**
# (short name: *Hospitality Env*,
# [repo](https://github.com/<your-username>/openenv-hospitality)).
#
# **Goal:** teach a small model (Qwen2.5-1.5B-Instruct) to act as a restaurant
# server — verify before acting, stay within authority, escalate when required,
# avoid unsafe allergy advice. We use GRPO (Group Relative Policy Optimization),
# a critic-free RL algorithm that fits comfortably on a single consumer GPU.
#
# **Why GRPO:** unlike PPO, GRPO has no value-function network. For each prompt
# it samples K completions, computes reward for each, and uses the group-relative
# advantage `(r_i - mean) / std` as the policy-gradient signal. Half the memory,
# no critic to train, good fit for small LLMs on small GPUs.

# %% [markdown]
# ## 1. Environment setup

# %%
# Install dependencies. On Colab, uncomment the whole block below.
# !pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q "trl==0.15.2" "transformers>=4.51.3,<=5.5.0" accelerate bitsandbytes  # 0.15.2 has GRPOTrainer without vllm/mergekit/llm-blender lazy imports
# mergekit / llm-blender not needed with trl 0.15.2 — they were lazy-imported starting trl 0.18+
# !pip install -q "pydantic>=2.10,<2.12"  # pydantic 2.12 breaks mergekit's MultislerpMergeTask (torch.Tensor field)
# !pip install -q openenv-core==0.2.3
# !pip install -q fastapi "uvicorn[standard]" httpx
# !pip install -q nest_asyncio  # needed because Colab already has a running event loop
#
# NOTE: After pip install finishes, do Runtime → Restart session before running
# subsequent cells, so the upgraded transformers/trl versions take effect.

# %%
# Allow asyncio.run_until_complete() to work inside Colab's existing event loop.
# Without this, every rollout call raises "This event loop is already running".
import nest_asyncio
nest_asyncio.apply()

# %%
# Clone and install the hospitality_env package.
# The pyproject.toml lives inside the hospitality_env/ subdirectory, so install from there.
# !git clone https://github.com/<your-username>/openenv-hospitality.git
# %cd openenv-hospitality
# !pip install --no-deps -e hospitality_env/  # --no-deps prevents re-upgrading pydantic via openenv-core → gradio/fastmcp

# %% [markdown]
# ## 2. Launch the hospitality env server in the background

# %%
import subprocess
import time
import httpx

# Start the FastAPI server as a background process.
server_proc = subprocess.Popen(
    [
        "python", "-m", "uvicorn",
        "hospitality_env.server.app:app",
        "--host", "127.0.0.1",
        "--port", "8000",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# Wait for health endpoint to respond.
for _ in range(30):
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
        if r.status_code == 200:
            print("Server up:", r.json())
            break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Server failed to start")

# %% [markdown]
# ## 3. Load Qwen2.5-1.5B with Unsloth 4-bit + LoRA

# %%
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # auto-detect (bf16 on Ampere+, fp16 elsewhere)
)

# Attach LoRA adapters to the attention and MLP projections.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

print("Model loaded. Trainable params:")
model.print_trainable_parameters()

# %% [markdown]
# ## 4. Episode rollout
#
# A single training example = one full episode on one task. We flatten the
# multi-turn conversation into a single `(prompt, completion, reward)` triple
# by treating the entire agent side of the conversation as the "completion"
# and the initial task description + first customer message as the "prompt."
#
# This is a v1 simplification — proper multi-turn RL would assign per-step
# rewards and credit-assign across turns. For a Colab-scale hackathon run,
# episode-level credit assignment is a reasonable first pass.

# %%
import json
import asyncio
from pathlib import Path
from hospitality_env.client import HospitalityEnv
from hospitality_env.models import HospitalityAction

# Load tasks for training / eval split.
TASKS_PATH = Path("hospitality_env/server/data/tasks.json")
with open(TASKS_PATH) as f:
    ALL_TASKS = json.load(f)

# Deterministic split: 100 train / 16 eval.
TRAIN_IDS = [t["id"] for t in ALL_TASKS[:100]]
EVAL_IDS = [t["id"] for t in ALL_TASKS[100:]]
print(f"Train: {len(TRAIN_IDS)}   Eval: {len(EVAL_IDS)}")


def build_prompt(system_message: str, customer_message: str) -> str:
    """Format the initial state as a chat-template prompt for Qwen."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": customer_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


async def rollout_once(task_id: str, generate_fn) -> dict:
    """Run one episode. `generate_fn(prompt) -> completion_text` drives the policy.

    Returns a dict with prompt, completion (agent's full text output), and
    total reward (sum of per-step + final rewards from the env).
    """
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    try:
        obs = await client.reset(task_id=task_id)
        system_message = obs.observation.system_message
        customer_message = obs.observation.customer_message

        prompt = build_prompt(system_message, customer_message)
        # v1 simplification: ask the model for a single text response;
        # do NOT attempt multi-turn tool-calling during training. We score
        # whatever the model outputs against the env by sending it as a
        # `message` action and reading the resulting reward.
        completion = generate_fn(prompt)

        step = await client.step(HospitalityAction(message=completion))
        total_reward = step.reward or 0.0

        # If the episode hasn't terminated after the first message, close
        # it cleanly so the env computes its final reward.
        for _ in range(3):
            if step.done:
                break
            step = await client.step(HospitalityAction(message="Goodbye."))
            total_reward += step.reward or 0.0

        return {
            "task_id": task_id,
            "prompt": prompt,
            "completion": completion,
            "reward": total_reward,
        }
    finally:
        await client.close()


# Helper: synchronous wrapper over the async rollout.
# Uses asyncio.run() which works in Colab after nest_asyncio.apply().
def rollout_sync(task_id: str, generate_fn) -> dict:
    return asyncio.run(rollout_once(task_id, generate_fn))


# %% [markdown]
# ## 5. Training dataset + reward function
#
# GRPOTrainer expects a HuggingFace dataset with a `prompt` column and a
# `reward_funcs` callable that scores `(prompts, completions) -> list[float]`.
# We pre-build prompts by resetting each training task once and caching the
# initial system + customer messages.

# %%
from datasets import Dataset


def build_train_dataset() -> Dataset:
    """Reset each train task once and cache its initial prompt."""
    rows = []
    for tid in TRAIN_IDS:
        async def _get():
            client = HospitalityEnv(base_url="http://127.0.0.1:8000")
            try:
                obs = await client.reset(task_id=tid)
                return obs.observation
            finally:
                await client.close()
        o = asyncio.run(_get())
        rows.append({
            "task_id": tid,
            "prompt": build_prompt(o.system_message, o.customer_message),
        })
    return Dataset.from_list(rows)


train_dataset = build_train_dataset()
print(f"Train dataset: {len(train_dataset)} prompts")


# %%
def reward_fn(prompts, completions, task_id=None, **kwargs):
    """GRPO reward function. For each (prompt, completion) pair, re-play
    the episode in the env and return the cumulative reward.

    GRPOTrainer will call this with a batch of completions sharing the same
    prompt (the K group rollouts) and average the advantage within each group.
    """
    rewards = []
    for tid, comp in zip(task_id, completions):
        async def _score():
            client = HospitalityEnv(base_url="http://127.0.0.1:8000")
            try:
                await client.reset(task_id=tid)
                step = await client.step(HospitalityAction(message=comp))
                r = step.reward or 0.0
                for _ in range(3):
                    if step.done:
                        break
                    step = await client.step(HospitalityAction(message="Goodbye."))
                    r += step.reward or 0.0
                return r
            finally:
                await client.close()
        rewards.append(asyncio.run(_score()))
    return rewards


# %% [markdown]
# ## 6. GRPO training

# %%
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./qwen_hospitality_grpo",
    # Core GRPO knobs
    num_generations=4,           # K = rollouts per prompt (group size)
    max_prompt_length=2048,
    max_completion_length=1024,
    temperature=0.9,             # need entropy to make GRPO work
    # Optimization
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,          # LoRA likes slightly higher; 1e-5 if overfitting
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_steps=100,               # Colab-scale budget; bump for real runs
    # Logging + saving
    logging_steps=1,
    save_steps=50,
    report_to="none",            # set to "wandb" to track externally
    # Precision
    bf16=True,
    # KL coefficient — tune if policy drifts too fast
    beta=0.04,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=train_dataset,
)

# %%
trainer.train()

# %% [markdown]
# ## 7. Save and push to Hugging Face Hub

# %%
OUTPUT_DIR = "./qwen_hospitality_lora"
HF_REPO_ID = "<your-hf-username>/qwen2.5-1.5b-hospitality-grpo"

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# %%
from huggingface_hub import HfApi, login
# login()  # run once, interactive token prompt

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Initial GRPO-trained LoRA adapter on hospitality env",
)

# %% [markdown]
# ## 8. Quick held-out eval
#
# Run the trained policy on the 16 eval tasks and compute mean reward.
# For full 116-task eval with completion-% analysis, use `baseline_eval.py`
# at the repo root.

# %%
FastLanguageModel.for_inference(model)


def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text


eval_rewards = []
for tid in EVAL_IDS:
    result = rollout_sync(tid, generate_text)
    eval_rewards.append(result["reward"])
    print(f"{tid}: reward={result['reward']:.3f}")

import statistics as st
print(f"\nMean eval reward:   {st.mean(eval_rewards):.3f}")
print(f"Median eval reward: {st.median(eval_rewards):.3f}")

# %% [markdown]
# ## 9. Cleanup

# %%
server_proc.terminate()
server_proc.wait(timeout=10)
print("Server stopped.")

# %% [markdown]
# ## Known v1 limitations
#
# - **Single-turn training shortcut.** We treat the entire agent response as
#   one completion, dropping true multi-turn credit assignment. A v2 run
#   would drive the full tool-calling loop inside the rollout and assign
#   per-step reward with discounting.
# - **Small step budget (100 steps).** Sufficient to show a training curve,
#   not sufficient to converge. A full run needs 500–2000 steps on a
#   better GPU.
# - **No tool-call schema in the completion.** The model emits free-form
#   text. Tool invocation parsing happens server-side by pattern matching.
#   A v2 would enforce structured JSON tool calls via format reward.
# - **Reward function re-runs the full episode per completion.** This is
#   slow (~seconds per rollout × K generations per batch). Caching the
#   (prompt, completion, reward) mapping would cut training time ~50%.
