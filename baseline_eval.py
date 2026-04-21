"""
Baseline evaluation: Claude Sonnet 4 plays the hospitality_env agent.

Connects to the deployed HF Space, samples N tasks, and for each task:
- Uses Claude as the "server" agent
- Feeds the agent observation (customer message + tool results) each turn
- Lets Claude choose: send message, call tool, or both
- Records reward, turn count, whether escalated, final status
- Writes per-task + summary CSV/JSON

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python baseline_eval.py --n 20 --model claude-sonnet-4-5
    python baseline_eval.py --n 20 --base-url http://127.0.0.1:8000   # local
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

# Disable WebSocket keepalive pings BEFORE importing the OpenEnv client.
# Claude API turns take 10-30s each — default 20s ping timeout would drop the WS.
# This works fine against a LOCAL server (no proxy in the middle).
import openenv.core.env_client as _env_client_mod
from websockets.asyncio.client import connect as _orig_ws_connect

def _patient_ws_connect(*args, **kwargs):
    kwargs["ping_interval"] = None
    kwargs["ping_timeout"] = None
    return _orig_ws_connect(*args, **kwargs)

_env_client_mod.ws_connect = _patient_ws_connect

from hospitality_env import HospitalityEnv, HospitalityAction
from agent_utils import SYSTEM_PROMPT, build_user_turn, parse_action, obs_to_dict

DEFAULT_LOCAL_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TURNS_SAFETY = 22  # hard ceiling (env caps at 20)


async def run_task(
    client: HospitalityEnv,
    anthropic: AsyncAnthropic,
    model: str,
    task_id: str,
) -> Dict[str, Any]:
    """Play one episode with Claude as the agent. Returns per-task record."""
    t_start = time.time()
    result = await client.reset(task_id=task_id)
    obs = obs_to_dict(result.observation)
    done = result.done

    history: List[Dict[str, str]] = []
    first = True
    last_reward: Optional[float] = None
    final_meta: Dict[str, Any] = {}
    escalated = False
    turns_used = 0
    error: Optional[str] = None

    for _ in range(MAX_TURNS_SAFETY):
        if done:
            final_meta = obs.get("metadata", {}) or {}
            break
        user_msg = build_user_turn(obs, first_turn=first)
        first = False
        history.append({"role": "user", "content": user_msg})

        try:
            resp = await anthropic.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=history,
            )
            assistant_text = resp.content[0].text if resp.content else ""
        except Exception as e:
            error = f"anthropic_error: {e}"
            break

        history.append({"role": "assistant", "content": assistant_text})
        action = parse_action(assistant_text)
        if action.tool_name and "escalate" in action.tool_name.lower():
            escalated = True

        try:
            result = await client.step(action)
        except Exception as e:
            error = f"env_step_error: {e}"
            break
        obs = obs_to_dict(result.observation)
        done = result.done
        if result.reward is not None:
            last_reward = result.reward
        turns_used = obs.get("turn_number", turns_used)
        if done:
            final_meta = obs.get("metadata", {}) or {}
            break

    return {
        "task_id": task_id,
        "reward": last_reward if last_reward is not None else 0.0,
        "turns": turns_used,
        "done": done,
        "escalated": escalated,
        "error": error,
        "reward_breakdown": final_meta,
        "duration_s": round(time.time() - t_start, 2),
    }


# ---------- Orchestration ----------

async def main_async(args):
    # Load task ids from local tasks.json (simpler than a schema endpoint)
    tasks_path = Path(__file__).parent / "hospitality_env" / "server" / "data" / "tasks.json"
    with open(tasks_path) as f:
        tasks = json.load(f)
    all_ids = [t["id"] for t in tasks]

    random.seed(args.seed)
    if args.task_ids:
        chosen = args.task_ids.split(",")
    else:
        chosen = random.sample(all_ids, min(args.n, len(all_ids)))

    print(f"Evaluating {len(chosen)} tasks with {args.model}")
    print(f"Env: {args.base_url}\n")

    anthropic = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    records: List[Dict[str, Any]] = []

    # Open a fresh WS connection per task to isolate failures.
    # ping_interval=None (monkey-patched above) keeps single-task WS alive
    # across long Claude API turns (only works against local server — HF
    # Space's reverse proxy has its own idle timeout we can't control).
    for i, tid in enumerate(chosen, 1):
        async with HospitalityEnv(base_url=args.base_url) as client:
            print(f"[{i}/{len(chosen)}] {tid} ...", end=" ", flush=True)
            try:
                rec = await run_task(client, anthropic, args.model, tid)
            except Exception as e:
                rec = {"task_id": tid, "error": f"fatal: {e}", "reward": 0.0,
                       "turns": 0, "done": False, "escalated": False,
                       "reward_breakdown": {}, "duration_s": 0}
            print(f"reward={rec['reward']:.3f} turns={rec['turns']} "
                  f"{'ERR' if rec['error'] else 'ok'}")
            records.append(rec)

    # Summary
    rewards = [r["reward"] for r in records]
    completed = [r for r in records if r["done"] and not r["error"]]
    errored = [r for r in records if r["error"]]
    print("\n" + "=" * 60)
    print(f"Summary  (n={len(records)}, model={args.model})")
    print("=" * 60)
    print(f"  mean reward    : {sum(rewards)/len(rewards):.3f}")
    print(f"  median reward  : {sorted(rewards)[len(rewards)//2]:.3f}")
    print(f"  completed      : {len(completed)}/{len(records)}")
    print(f"  errored        : {len(errored)}")
    print(f"  mean turns     : {sum(r['turns'] for r in records)/len(records):.1f}")
    print(f"  escalation rate: {sum(r['escalated'] for r in records)/len(records)*100:.1f}%")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"baseline_{args.model}_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "records": records}, f, indent=2)
    print(f"\nSaved -> {out_json}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Number of tasks to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_LOCAL_URL)
    p.add_argument("--task-ids", default="", help="Comma-separated task ids (overrides --n)")
    p.add_argument("--out-dir", default="eval_results")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
