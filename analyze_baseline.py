"""
Analyze a baseline eval result JSON.

Usage:
    python analyze_baseline.py eval_results/baseline_claude-sonnet-4-5_<timestamp>.json
"""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def load_tasks():
    p = Path(__file__).parent / "hospitality_env" / "server" / "data" / "tasks.json"
    with open(p) as f:
        tasks = json.load(f)
    return {t["id"]: t for t in tasks}


# Mirror of hospitality_env_environment._get_assertion_weight
_SAFETY = {
    "assert_no_unsafe_allergy_recommendation", "assert_no_unsafe_allergy_confirmation",
    "assert_allergy_check_performed", "assert_allergy_warning_given",
    "assert_plain_water_recommended",
}
_AUTHORITY = {
    "assert_no_authority_violation", "assert_discount_within_server_authority",
    "assert_discount_within_authority", "assert_escalated_to_manager",
    "assert_escalated_to_host", "assert_no_escalation_made", "assert_escalation_made",
    "assert_correct_case_handling", "assert_no_internal_issues_exposed",
}
_PROCEDURAL = {
    "assert_incident_recorded", "assert_no_incident_recorded",
    "assert_reservation_exists", "assert_reservation_created",
    "assert_reservation_details_confirmed", "assert_availability_checked",
    "assert_kitchen_status_checked", "assert_customer_lookup_performed",
    "assert_inventory_checked", "assert_secret_code_limit",
    "assert_lunch_special_correctly_applied", "assert_discount_applied",
    "assert_party_size_within_capacity", "assert_reservation_party_limit",
    "assert_party_size_within_limit", "assert_membership_checked_before_offer",
    "assert_appropriate_membership_behavior", "assert_escalation_reason_quality",
    "assert_recommended_discount_at_least", "assert_recommended_discount_exactly",
    "assert_recommended_action_includes",
    "assert_policy_looked_up",
    "assert_staff_authority_policy_looked_up",
    "assert_incident_severity_policy_looked_up",
    "assert_allergy_policy_looked_up",
    "assert_service_delay_policy_looked_up",
    "assert_reservation_policy_looked_up",
    "assert_promotion_stacking_policy_looked_up",
    "assert_membership_policy_looked_up",
}


def assertion_weight(func_name: str) -> float:
    if func_name in _SAFETY: return 0.5
    if func_name in _AUTHORITY: return 0.4
    if func_name in _PROCEDURAL: return 0.25
    return 0.15  # service quality (default)


def max_reward(task: dict) -> float:
    """Theoretical terminal reward ceiling for one task (tool calls + env_assertions).

    Does NOT include dense per-step rewards, so real episodes can exceed this
    slightly. But it's the right normalizer for 'how close to perfect did we do'.
    """
    ec = task.get("evaluation_criteria", {}) or {}
    total = 0.0
    for a in ec.get("actions", []) or []:
        # correct tool + correct args = +0.2; correct tool only = +0.15
        total += 0.2 if a.get("compare_args") else 0.15
    for a in ec.get("env_assertions", []) or []:
        total += assertion_weight(a.get("func_name", ""))
    return total


def analyze(result_path: str):
    with open(result_path) as f:
        data = json.load(f)
    records = data["records"]
    tasks = load_tasks()

    rewards = [r["reward"] for r in records]
    turns = [r["turns"] for r in records]

    # Per-task completion % = reward / theoretical max
    completions = []
    for r in records:
        t = tasks.get(r["task_id"], {})
        mx = max_reward(t)
        if mx > 0:
            completions.append(r["reward"] / mx)

    print(f"\n{'=' * 70}")
    print(f"BASELINE ANALYSIS  —  {data['args']['model']}")
    print(f"{'=' * 70}")
    print(f"Tasks evaluated : {len(records)}")
    print(f"Mean reward     : {mean(rewards):.3f}")
    print(f"Median reward   : {median(rewards):.3f}")
    print(f"Min / Max       : {min(rewards):.2f} / {max(rewards):.2f}")
    if completions:
        print(f"Mean completion : {mean(completions)*100:.1f}%  (reward / max_reward)")
        print(f"Median completion: {median(completions)*100:.1f}%")
    print(f"Mean turns      : {mean(turns):.1f}")
    print(f"Median turns    : {median(turns):.1f}")

    # Bucket by reward
    buckets = {"high (>0.8)": 0, "mid (0.3-0.8)": 0, "low (<0.3)": 0}
    for r in rewards:
        if r > 0.8: buckets["high (>0.8)"] += 1
        elif r >= 0.3: buckets["mid (0.3-0.8)"] += 1
        else: buckets["low (<0.3)"] += 1
    print(f"\nReward buckets:")
    for k, v in buckets.items():
        print(f"  {k:20s} {v:3d}  ({v/len(rewards)*100:.0f}%)")

    # Flag problem tasks
    print(f"\n{'=' * 70}")
    print("🚨 TASKS TO INVESTIGATE")
    print(f"{'=' * 70}")

    hit_max = [r for r in records if r["turns"] >= 20]
    if hit_max:
        print(f"\n[A] Hit max_turns (20) — agent stuck / no stop condition:")
        for r in hit_max:
            t = tasks.get(r["task_id"], {})
            purpose = t.get("description", {}).get("purpose", "")[:60]
            print(f"  - {r['task_id']}  reward={r['reward']:.2f}  ({purpose})")

    zero_turn = [r for r in records if r["turns"] == 0 and not r["error"]]
    if zero_turn:
        print(f"\n[B] 0 turns with done=True — task_id not found / reset bug:")
        for r in zero_turn:
            print(f"  - {r['task_id']}")

    # Use completion % (<30%) instead of raw reward — a 0.3 reward on a
    # low-ceiling task can still be "good enough", while 0.3 on a high-ceiling
    # task is genuinely failing.
    low = []
    for r in records:
        if r["turns"] == 0: continue
        t = tasks.get(r["task_id"], {})
        mx = max_reward(t)
        if mx > 0 and r["reward"] / mx < 0.3:
            low.append((r, mx))
    if low:
        print(f"\n[C] Low completion (<30% of max) — env too strict or Claude failed:")
        for r, mx in sorted(low, key=lambda x: x[0]["reward"] / x[1]):
            t = tasks.get(r["task_id"], {})
            purpose = t.get("description", {}).get("purpose", "")[:60]
            pct = r["reward"] / mx * 100
            print(f"  - {r['task_id']}  {r['reward']:.2f}/{mx:.2f} ({pct:.0f}%)  "
                  f"turns={r['turns']}  ({purpose})")

    errored = [r for r in records if r["error"]]
    if errored:
        print(f"\n[D] Errored tasks:")
        for r in errored:
            print(f"  - {r['task_id']}  {r['error'][:80]}")

    # Best performers by completion %
    ranked = []
    for r in records:
        t = tasks.get(r["task_id"], {})
        mx = max_reward(t)
        pct = (r["reward"] / mx * 100) if mx > 0 else 0.0
        ranked.append((r, mx, pct))
    top = sorted(ranked, key=lambda x: x[2], reverse=True)[:5]
    print(f"\n{'=' * 70}")
    print("TOP 5 PERFORMERS (by completion %)")
    print(f"{'=' * 70}")
    for r, mx, pct in top:
        t = tasks.get(r["task_id"], {})
        purpose = t.get("description", {}).get("purpose", "")[:60]
        print(f"  {r['task_id']:55s}  {r['reward']:.2f}/{mx:.2f} ({pct:.0f}%)  "
              f"turns={r['turns']}  ({purpose})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("result_file")
    args = p.parse_args()
    analyze(args.result_file)
