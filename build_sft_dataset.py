"""
Build an SFT dataset from a baseline_eval.py run with --save-transcripts.

Filters transcripts by reward threshold, converts to ChatML-style messages
suitable for Unsloth / trl SFTTrainer on Qwen2.5-Instruct, and writes JSONL.

Usage:
    python build_sft_dataset.py \
        --input eval_results/baseline_claude-sonnet-4-5_20260420_XXXXXX.json \
        --min-reward 0.7 \
        --output sft_data/hospitality_sft.jsonl

Each output line:
    {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "=== TASK CONTEXT === ..."},
        {"role": "assistant", "content": "{\"message\": ..., \"tool_name\": ...}"},
        {"role": "user",      "content": "=== CUSTOMER SAYS === ..."},
        ...
    ],
     "task_id": "...", "reward": 0.83, "turns": 5}
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="baseline JSON with transcripts")
    p.add_argument("--min-reward", type=float, default=0.7)
    p.add_argument("--output", default="sft_data/hospitality_sft.jsonl")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap number of records (after filtering)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    records = data["records"]

    # Filter: reward high, no error, has transcript
    good = [
        r for r in records
        if r.get("reward", 0) >= args.min_reward
        and not r.get("error")
        and r.get("messages")
        and r.get("system_prompt")
    ]
    if args.max_examples:
        good = good[: args.max_examples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    total_turns = 0
    with open(out_path, "w") as f:
        for r in good:
            msgs = [{"role": "system", "content": r["system_prompt"]}]
            msgs.extend(r["messages"])   # already alternating user/assistant
            # Sanity: must end on assistant for SFT loss to include final reply
            if msgs[-1]["role"] != "assistant":
                continue
            line = {
                "messages": msgs,
                "task_id": r["task_id"],
                "reward": r["reward"],
                "turns": r["turns"],
            }
            f.write(json.dumps(line) + "\n")
            n_written += 1
            total_turns += r["turns"]

    print(f"Input:  {args.input}")
    print(f"  total records:     {len(records)}")
    print(f"  after reward>={args.min_reward}: {len(good)}")
    print(f"  written (valid end): {n_written}")
    if n_written:
        print(f"  mean turns/example:  {total_turns/n_written:.1f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
