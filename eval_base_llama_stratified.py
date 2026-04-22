"""
Cross-family base control: eval BASE Meta-Llama-3.1-8B-Instruct on stratified.

Prior data on same stratified eval set (20 tasks, 10 categories):
  Claude Sonnet 4.5:            +0.314
  7B Qwen v1 (SFT, 37 traj):    +0.101
  7B Qwen base (no SFT):        +0.101  (per-task IDENTICAL to v1)
  14B Qwen base (no SFT):       +0.068

Two Qwen data points clustered near Claude's 1/3. Test (b) a Qwen-family-
specific oddity: run Meta's Llama 3.1 8B (different architecture family,
different training data) through the exact same inference stack.

  If Llama-3.1-8B base >> +0.15: Qwen family has some JSON/tool-call bias
      that suppressed performance. Story pivots to model-family matters.
  If Llama-3.1-8B base ~= +0.10: triple-family null. Env-data bottleneck
      becomes the locked conclusion. Blog thesis is ironclad.

15-25 min on H100, ~$1.

NB on prompting: Llama 3.1 uses a different chat template than Qwen.
tokenizer.apply_chat_template handles this automatically (both models
ship their own template). SYSTEM_PROMPT_V2 content is identical.

Usage:
  cd /home/ubuntu/openenv-hospitality
  python eval_base_llama_stratified.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 8192

REPO_ROOT = "/home/ubuntu/openenv-hospitality"
EVAL_IDS_PATH = f"{REPO_ROOT}/sft_data/stratified_eval_ids.json"
TASKS_PATH = f"{REPO_ROOT}/hospitality_env/server/data/tasks.json"
BASELINE_PATH = f"{REPO_ROOT}/eval_results/baseline_claude-sonnet-4-5_20260421_002809.json"
EVAL_OUT = "/home/ubuntu/hospitality/eval/eval_heldout_base_llama_stratified.json"
os.makedirs(os.path.dirname(EVAL_OUT), exist_ok=True)

SEED = 42


# -------- start env server --------
import sys, subprocess, time, httpx

print("Starting env server ...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "hospitality_env.server.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT,
)
for _ in range(30):
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
        if r.status_code == 200:
            print("Server up:", r.json()); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Server failed to come up.")


# -------- load BASE model (no LoRA, no adapter) --------
from unsloth import FastLanguageModel
from transformers import set_seed
import torch, random, numpy as np

set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

print(f"\nLoading BASE model {BASE_MODEL} (no SFT, no adapter) ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
# NOTE: do NOT call get_peft_model. We want the raw instruct-tuned base,
# nothing else.
FastLanguageModel.for_inference(model)
print("Base model loaded, inference mode. Trainable params = 0 (no adapter).")


# -------- agent wrapper (verbatim from train_lambda.py §6) --------
from hospitality_env import HospitalityAction
from hospitality_env.client import HospitalityEnv
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
            parts.append(f"\n=== TOOL SCHEMAS ===\n{_json.dumps(obs['tool_schemas'], indent=2)}")
    if obs.get("customer_message"):
        parts.append(f"\n=== CUSTOMER SAYS ===\n{obs['customer_message']}")
    if obs.get("tool_result"):
        parts.append(f"\n=== LAST TOOL RESULT ===\n{obs['tool_result']}")
    if obs.get("tool_error"):
        parts.append(f"\n=== LAST TOOL ERROR ===\n{obs['tool_error']}")
    parts.append(
        f"\n[turn {obs.get('turn_number', 0)}/{obs.get('max_turns', 20)}] Respond with JSON."
    )
    return "\n".join(parts)


def parse_action(text: str) -> HospitalityAction:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1:
        return HospitalityAction(message=text[:500])
    try:
        data = _json.loads(text[start:end + 1])
    except _json.JSONDecodeError:
        return HospitalityAction(message=text[:500])
    return HospitalityAction(
        message=data.get("message", "") or "",
        tool_name=data.get("tool_name", "") or "",
        tool_args=data.get("tool_args", {}) or {},
    )


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


async def rollout_v3(tid: str, max_turns: int = 8):
    client = HospitalityEnv(base_url="http://127.0.0.1:8000")
    total_reward, turns, tool_calls = 0.0, 0, 0
    no_tool_streak = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2}]
    trace = []
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
            if step.done:
                break
            obs = step.observation
        return {"task_id": tid, "reward": total_reward, "turns": turns,
                "tool_calls": tool_calls, "trace": trace}
    except Exception as e:
        return {"task_id": tid, "reward": None, "turns": turns,
                "tool_calls": tool_calls, "error": str(e)[:200], "trace": trace}
    finally:
        await client.close()


# -------- run eval --------
import json, asyncio, statistics as st
from collections import Counter, defaultdict

EVAL_IDS = json.load(open(EVAL_IDS_PATH))
ALL_TASKS = json.load(open(TASKS_PATH))
TID_TO_CAT = {t["id"]: t["description"]["category"] for t in ALL_TASKS}

print(f"\nStratified eval set: {len(EVAL_IDS)} tasks, "
      f"{len(set(TID_TO_CAT[tid] for tid in EVAL_IDS))} categories")
for cat, n in sorted(Counter(TID_TO_CAT[tid] for tid in EVAL_IDS).items()):
    print(f"  {cat:24s} {n}")

print(f"\nRunning BASE {BASE_MODEL} on {len(EVAL_IDS)} stratified tasks ...")
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


# -------- summary --------
valid = [r for r in results if r["reward"] is not None]
rewards = [r["reward"] for r in valid]
mean_reward = st.mean(rewards)

print(f"\n=== BASE {BASE_MODEL} on STRATIFIED eval (NO SFT) ===")
print(f"Mean:     {mean_reward:+.3f}")
print(f"Median:   {st.median(rewards):+.3f}")
print(f"Min/Max:  {min(rewards):+.3f} / {max(rewards):+.3f}")
print(f"Mean turns: {st.mean([r['turns'] for r in valid]):.2f}")
print(f"\nContext — all on SAME stratified 20-task eval:")
print(f"  Claude Sonnet 4.5:            +0.314")
print(f"  Qwen 7B v1 (SFT, 37 traj):    +0.101")
print(f"  Qwen 7B v2 (SFT, 50 traj):    +0.133")
print(f"  Qwen 7B v3 (SFT, 40 traj):    +0.104")
print(f"  Qwen 7B BASE (no SFT):        +0.101")
print(f"  Qwen 14B BASE (no SFT):       +0.068")
print(f"  Llama 3.1 8B BASE (no SFT):   {mean_reward:+.3f}  ← THIS RUN")
print()
_delta_vs_qwen7b = mean_reward - 0.101
print(f"Llama 8B vs Qwen 7B base: Δ = {_delta_vs_qwen7b:+.3f}")
if mean_reward >= 0.15:
    print("  → Qwen-family-specific failure. Try Llama-family SFT instead.")
elif abs(_delta_vs_qwen7b) < 0.05:
    print("  → TRIPLE-FAMILY NULL CONFIRMED (Qwen-7B ≈ Qwen-14B ≈ Llama-8B).")
    print("    Bottleneck is ENV DATA, not model family or capacity.")
    print("    Ship the null-result story.")
else:
    print("  → Mixed signal — compare per-category breakdown below.")

by_cat = defaultdict(list)
for r in valid:
    by_cat[r["category"]].append(r["reward"])

claude_records = {r["task_id"]: r["reward"]
                  for r in json.load(open(BASELINE_PATH))["records"]}
claude_by_cat = defaultdict(list)
for tid in EVAL_IDS:
    claude_by_cat[TID_TO_CAT[tid]].append(claude_records[tid])

print(f"\n=== Per-category (base vs Claude) ===")
print(f"{'category':24s} {'n':>3s}  {'base':>7s}  {'Claude':>7s}  {'Δ':>7s}  {'verdict':>10s}")
print("-" * 80)
cat_rows = []
for cat in sorted(by_cat.keys()):
    ours = st.mean(by_cat[cat])
    cla = st.mean(claude_by_cat[cat]) if cat in claude_by_cat else float("nan")
    diff = ours - cla
    verdict = "BEAT" if diff > 0.05 else "MATCH" if abs(diff) <= 0.05 else "LOSE"
    cat_rows.append((cat, len(by_cat[cat]), ours, cla, diff, verdict))
    print(f"{cat:24s} {len(by_cat[cat]):>3d}  "
          f"{ours:>+7.3f}  {cla:>+7.3f}  {diff:>+7.3f}  {verdict:>10s}")

# Freeze count (diagnostic for "does SFT actually change behavior vs base?")
freeze_count = sum(1 for r in valid if r["turns"] == 8 and r["tool_calls"] == 0)
print(f"\nFreeze rate (turns=8, tools=0): {freeze_count}/{len(valid)}")
print(f"  v1:   6/20")
print(f"  v3:  11/20")


# -------- save --------
with open(EVAL_OUT, "w") as f:
    json.dump({
        "model": BASE_MODEL,
        "sft": False,
        "eval_set": "stratified_v3",
        "n_eval": len(EVAL_IDS),
        "claude_on_same_eval": 0.314,
        "freeze_count": freeze_count,
        "results": results,
        "per_category": [
            {"category": c, "n": n, "base": o, "claude": cl, "delta": d, "verdict": v}
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

try:
    server_proc.terminate(); server_proc.wait(timeout=5)
except Exception:
    pass
print("Done.")
