"""
Compare two baseline eval result JSONs — "v0 vs v1" style ablation-style report.

Usage:
    python compare_baselines.py <v0.json> <v1.json> \\
        [--v0-tasks hospitality_env/server/data/tasks.json.bak] \\
        [--v1-tasks hospitality_env/server/data/tasks.json]

Notes:
- v0 = "before" baseline (old environment)
- v1 = "after" baseline (new environment)
- Each baseline's completion% uses its OWN tasks.json for max_reward,
  so completion% is comparable even though raw max ceilings differ.
- This is NOT a clean ablation (multiple things changed together) —
  the output is directional evidence, not per-component attribution.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

# Mirror of analyze_baseline.py's weight map
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
    return 0.15


def max_reward_of(task: dict) -> float:
    ec = task.get("evaluation_criteria", {}) or {}
    total = 0.0
    for a in ec.get("actions", []) or []:
        total += 0.2 if a.get("compare_args") else 0.15
    for a in ec.get("env_assertions", []) or []:
        total += assertion_weight(a.get("func_name", ""))
    return total


def load_tasks(path: str) -> dict:
    with open(path) as f:
        return {t["id"]: t for t in json.load(f)}


def summarize(records, tasks_by_id):
    """Return dict of aggregate metrics for one baseline run."""
    rewards = [r["reward"] for r in records]
    turns = [r["turns"] for r in records]
    completions = []
    per_category = defaultdict(list)  # category -> list of completion %
    for r in records:
        t = tasks_by_id.get(r["task_id"], {})
        mx = max_reward_of(t)
        comp = (r["reward"] / mx) if mx > 0 else 0.0
        completions.append(comp)
        cat = t.get("description", {}).get("category", "unknown")
        per_category[cat].append(comp)

    buckets = {"high": 0, "mid": 0, "low": 0}
    for r in rewards:
        if r > 0.8:
            buckets["high"] += 1
        elif r >= 0.3:
            buckets["mid"] += 1
        else:
            buckets["low"] += 1

    return {
        "n": len(records),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "median_reward": median(rewards) if rewards else 0.0,
        "mean_completion": mean(completions) if completions else 0.0,
        "median_completion": median(completions) if completions else 0.0,
        "mean_turns": mean(turns) if turns else 0.0,
        "median_turns": median(turns) if turns else 0.0,
        "buckets": buckets,
        "per_category": {k: mean(v) for k, v in per_category.items() if v},
        "records_by_id": {r["task_id"]: (r, max_reward_of(tasks_by_id.get(r["task_id"], {})))
                          for r in records},
    }


def fmt_delta(v0, v1, pct=False, invert=False):
    """Format delta with arrow. `invert=True` means lower is better (e.g., turns)."""
    d = v1 - v0
    if pct:
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
        return f"{v0*100:5.1f}% → {v1*100:5.1f}%  ({arrow}{abs(d)*100:.1f}pp)"
    else:
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
        good = (d > 0) ^ invert
        marker = "✓" if good and abs(d) > 0.01 else (" " if abs(d) < 0.01 else "!")
        return f"{v0:6.3f} → {v1:6.3f}  ({arrow}{abs(d):.3f}) {marker}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("v0", help="Old baseline JSON")
    ap.add_argument("v1", help="New baseline JSON")
    ap.add_argument("--v0-tasks", default="hospitality_env/server/data/tasks.json.bak",
                    help="tasks.json used by the v0 run (for max_reward)")
    ap.add_argument("--v1-tasks", default="hospitality_env/server/data/tasks.json",
                    help="tasks.json used by the v1 run (for max_reward)")
    args = ap.parse_args()

    with open(args.v0) as f:
        v0_data = json.load(f)
    with open(args.v1) as f:
        v1_data = json.load(f)

    v0_tasks = load_tasks(args.v0_tasks)
    v1_tasks = load_tasks(args.v1_tasks)

    v0 = summarize(v0_data["records"], v0_tasks)
    v1 = summarize(v1_data["records"], v1_tasks)

    print("=" * 72)
    print("BASELINE DELTA  —  v0 vs v1")
    print("=" * 72)
    print(f"  v0: {Path(args.v0).name}  (tasks: {Path(args.v0_tasks).name})")
    print(f"  v1: {Path(args.v1).name}  (tasks: {Path(args.v1_tasks).name})")
    print(f"  n(v0) = {v0['n']}   n(v1) = {v1['n']}")
    print()

    print("OVERALL")
    print(f"  Mean reward       {fmt_delta(v0['mean_reward'], v1['mean_reward'])}")
    print(f"  Median reward     {fmt_delta(v0['median_reward'], v1['median_reward'])}")
    print(f"  Mean completion   {fmt_delta(v0['mean_completion'], v1['mean_completion'], pct=True)}")
    print(f"  Median completion {fmt_delta(v0['median_completion'], v1['median_completion'], pct=True)}")
    print(f"  Mean turns        {fmt_delta(v0['mean_turns'], v1['mean_turns'], invert=True)}")
    print(f"  Median turns      {fmt_delta(v0['median_turns'], v1['median_turns'], invert=True)}")
    print()

    print("REWARD BUCKETS (raw reward distribution)")
    for k in ("high", "mid", "low"):
        p0 = v0["buckets"][k] / v0["n"] * 100
        p1 = v1["buckets"][k] / v1["n"] * 100
        label = {"high": "high (>0.8)   ", "mid": "mid  (0.3-0.8)", "low": "low  (<0.3)  "}[k]
        print(f"  {label}  {v0['buckets'][k]:3d} ({p0:4.1f}%) → {v1['buckets'][k]:3d} ({p1:4.1f}%)")
    print()
    print("  Note: raw reward buckets not directly comparable — v1 has higher max ceilings.")
    print("  Use completion-% categories below for fair comparison.")
    print()

    print("PER-CATEGORY MEAN COMPLETION %")
    cats = sorted(set(v0["per_category"].keys()) | set(v1["per_category"].keys()))
    for cat in cats:
        c0 = v0["per_category"].get(cat, None)
        c1 = v1["per_category"].get(cat, None)
        if c0 is None or c1 is None:
            continue
        d = (c1 - c0) * 100
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
        marker = "✓" if d > 2 else ("!" if d < -2 else " ")
        print(f"  {cat:24s}  {c0*100:5.1f}% → {c1*100:5.1f}%  ({arrow}{d:+5.1f}pp) {marker}")
    print()

    print("TOP 10 TASK-LEVEL IMPROVEMENTS (completion% delta)")
    deltas = []
    common_ids = set(v0["records_by_id"].keys()) & set(v1["records_by_id"].keys())
    for tid in common_ids:
        r0, mx0 = v0["records_by_id"][tid]
        r1, mx1 = v1["records_by_id"][tid]
        c0 = r0["reward"] / mx0 if mx0 > 0 else 0
        c1 = r1["reward"] / mx1 if mx1 > 0 else 0
        deltas.append((tid, c0, c1, c1 - c0))
    deltas.sort(key=lambda x: x[3], reverse=True)
    for tid, c0, c1, d in deltas[:10]:
        print(f"  {tid:55s}  {c0*100:5.1f}% → {c1*100:5.1f}%  (↑{d*100:+5.1f}pp)")
    print()

    print("TOP 10 TASK-LEVEL REGRESSIONS")
    for tid, c0, c1, d in deltas[-10:][::-1]:
        if d >= 0:
            continue
        print(f"  {tid:55s}  {c0*100:5.1f}% → {c1*100:5.1f}%  (↓{d*100:+5.1f}pp)")
    print()

    # Calibration: how many tasks got better vs worse
    better = sum(1 for *_, d in deltas if d > 0.05)
    worse = sum(1 for *_, d in deltas if d < -0.05)
    same = len(deltas) - better - worse
    print("CALIBRATION")
    print(f"  Tasks improved >5pp  : {better} / {len(deltas)} ({better/len(deltas)*100:.0f}%)")
    print(f"  Tasks regressed >5pp : {worse} / {len(deltas)} ({worse/len(deltas)*100:.0f}%)")
    print(f"  Tasks roughly same   : {same} / {len(deltas)} ({same/len(deltas)*100:.0f}%)")


if __name__ == "__main__":
    main()
