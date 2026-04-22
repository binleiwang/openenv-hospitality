"""
Build stratified train/eval split + SFT pool for Phase 5 (v3).

Rationale (from Phase 4 postmortem):
  Threshold-based global filter on Claude baseline produces a
  training pool that's biased toward Claude's strong categories
  (server_misc, food_safety, food_issue) and contains ZERO examples
  for categories Claude struggles with (host_seating has 0 tasks
  above reward 0.5). The SFT model never sees those decision
  patterns in training, so it fails on them at eval time.

Fix:
  1. 80/20 train/eval split WITHIN EACH CATEGORY (stratified holdout)
     → eval set covers all 10 categories, not a random 15-task slice
  2. Per-category top-K SFT pool (not global threshold)
     → every category contributes at least some demonstrations,
       even if Claude's best is mediocre there

Output:
  sft_data/stratified_train_ids.json   # train task IDs (~95)
  sft_data/stratified_eval_ids.json    # eval task IDs (~21)
  sft_data/hospitality_sft_strat.jsonl  # SFT training data
"""

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def build(
    tasks_path: str,
    baseline_path: str,
    out_dir: str = "sft_data",
    eval_frac: float = 0.20,
    eval_min: int = 1,
    eval_max: int = 4,
    sft_per_cat_frac: float = 0.5,
    sft_per_cat_min: int = 2,
    sft_per_cat_max: int = 8,
    seed: int = 42,
) -> None:
    random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = json.load(open(tasks_path))
    tid_to_cat = {t["id"]: t["description"]["category"] for t in tasks}

    baseline = json.load(open(baseline_path))
    records = baseline["records"]
    tid_to_record = {r["task_id"]: r for r in records}

    # Group tasks by category
    by_cat: dict[str, list[str]] = defaultdict(list)
    for tid, cat in tid_to_cat.items():
        by_cat[cat].append(tid)

    # --- Stratified 80/20 split ---
    train_ids: list[str] = []
    eval_ids: list[str] = []

    print(f"\n{'Category':24s} {'total':>5s} {'train':>5s} {'eval':>4s}  "
          f"{'Claude_mean':>11s}  {'eval_ids':>8s}")
    print("-" * 80)

    for cat in sorted(by_cat.keys()):
        cat_tids = by_cat[cat]
        if len(cat_tids) == 1:
            # Singleton goes to train; acknowledge no eval coverage
            train_ids.extend(cat_tids)
            claude_mean = tid_to_record[cat_tids[0]]["reward"]
            print(f"{cat:24s} {len(cat_tids):>5d} {len(cat_tids):>5d} {0:>4d}"
                  f"  {claude_mean:>+11.3f}  (singleton, no eval)")
            continue

        # Sort by Claude reward (descending) so best-of-best goes to SFT
        # Use reward-based sort for reproducibility independent of random shuffle
        cat_tids_sorted_by_reward = sorted(
            cat_tids, key=lambda t: tid_to_record[t]["reward"], reverse=True
        )
        # Now random.shuffle on a copy to pick eval set (so eval is random,
        # but SFT pool below uses reward-sorted order)
        shuffled = list(cat_tids)
        random.shuffle(shuffled)

        n_eval = max(eval_min, min(eval_max, round(len(cat_tids) * eval_frac)))
        cat_eval = shuffled[:n_eval]
        cat_train = [t for t in cat_tids if t not in set(cat_eval)]

        train_ids.extend(cat_train)
        eval_ids.extend(cat_eval)

        claude_mean = st.mean(tid_to_record[t]["reward"] for t in cat_tids)
        eval_mean = st.mean(tid_to_record[t]["reward"] for t in cat_eval)
        print(f"{cat:24s} {len(cat_tids):>5d} {len(cat_train):>5d} {n_eval:>4d}"
              f"  {claude_mean:>+11.3f}  "
              f"(eval Claude: {eval_mean:+.3f})")

    print("-" * 80)
    print(f"{'TOTAL':24s} {len(train_ids)+len(eval_ids):>5d} "
          f"{len(train_ids):>5d} {len(eval_ids):>4d}")
    claude_overall = st.mean(r["reward"] for r in records)
    claude_eval_mean = st.mean(tid_to_record[t]["reward"] for t in eval_ids)
    print(f"\nClaude overall (116 tasks):           {claude_overall:+.3f}")
    print(f"Claude on stratified eval ({len(eval_ids)} tasks): {claude_eval_mean:+.3f}")
    print(f"(Reference target — v3 should approach this)")

    # --- Build SFT pool: per-category top-K from TRAIN tasks only ---
    sft_rows = []
    sft_stats = []
    print(f"\n{'Category':24s} {'train':>5s} {'sft_k':>5s}  {'kept_reward_range':>18s}")
    print("-" * 80)

    for cat in sorted(by_cat.keys()):
        cat_train_ids = [t for t in by_cat[cat] if t in set(train_ids)]
        if not cat_train_ids:
            continue

        # Rank by Claude reward within this category
        cat_train_ranked = sorted(
            cat_train_ids, key=lambda t: tid_to_record[t]["reward"], reverse=True
        )

        k = max(
            sft_per_cat_min,
            min(sft_per_cat_max, round(len(cat_train_ids) * sft_per_cat_frac)),
        )
        k = min(k, len(cat_train_ids))
        kept = cat_train_ranked[:k]

        for tid in kept:
            rec = tid_to_record[tid]
            if rec.get("error"):
                continue
            sft_rows.append({
                "task_id": tid,
                "category": cat,
                "reward": rec["reward"],
                "turns": rec["turns"],
                "messages": rec["messages"],
            })

        kept_rewards = [tid_to_record[t]["reward"] for t in kept]
        sft_stats.append({
            "category": cat,
            "n_kept": len(kept),
            "min": min(kept_rewards), "max": max(kept_rewards),
            "mean": st.mean(kept_rewards),
        })
        print(f"{cat:24s} {len(cat_train_ids):>5d} {len(kept):>5d}  "
              f"[{min(kept_rewards):>+.2f} .. {max(kept_rewards):>+.2f}]  "
              f"mean {st.mean(kept_rewards):+.3f}")

    print("-" * 80)
    print(f"{'SFT TOTAL':24s}       {len(sft_rows):>5d}")
    print(f"Unique categories in pool: {len(set(r['category'] for r in sft_rows))} "
          f"(target: all {len(by_cat)})")

    # --- Write outputs ---
    with open(out / "stratified_train_ids.json", "w") as f:
        json.dump(train_ids, f, indent=2)
    with open(out / "stratified_eval_ids.json", "w") as f:
        json.dump(eval_ids, f, indent=2)
    with open(out / "hospitality_sft_strat.jsonl", "w") as f:
        for row in sft_rows:
            f.write(json.dumps(row) + "\n")

    # Also write a manifest with key numbers for README / blog later
    manifest = {
        "seed": seed,
        "n_tasks_total": len(tasks),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "n_sft_examples": len(sft_rows),
        "claude_overall_mean": claude_overall,
        "claude_on_stratified_eval": claude_eval_mean,
        "eval_frac": eval_frac,
        "sft_per_cat_frac": sft_per_cat_frac,
        "sft_per_cat_min": sft_per_cat_min,
        "sft_per_cat_max": sft_per_cat_max,
        "categories": sft_stats,
    }
    with open(out / "stratified_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote:")
    print(f"  {out / 'stratified_train_ids.json'}  ({len(train_ids)} task IDs)")
    print(f"  {out / 'stratified_eval_ids.json'}   ({len(eval_ids)} task IDs)")
    print(f"  {out / 'hospitality_sft_strat.jsonl'} ({len(sft_rows)} SFT examples)")
    print(f"  {out / 'stratified_manifest.json'}   (summary)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="hospitality_env/server/data/tasks.json")
    ap.add_argument("--baseline",
                    default="eval_results/baseline_claude-sonnet-4-5_20260421_002809.json")
    ap.add_argument("--out", default="sft_data")
    ap.add_argument("--eval-frac", type=float, default=0.20)
    ap.add_argument("--sft-per-cat-frac", type=float, default=0.5)
    ap.add_argument("--sft-per-cat-min", type=int, default=2)
    ap.add_argument("--sft-per-cat-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build(
        tasks_path=args.tasks,
        baseline_path=args.baseline,
        out_dir=args.out,
        eval_frac=args.eval_frac,
        sft_per_cat_frac=args.sft_per_cat_frac,
        sft_per_cat_min=args.sft_per_cat_min,
        sft_per_cat_max=args.sft_per_cat_max,
        seed=args.seed,
    )
