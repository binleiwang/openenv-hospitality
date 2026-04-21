# Hospitality RL Environment: A Hot Pot Restaurant Simulation

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment that simulates a full-service hot pot restaurant ("Berkeley Hot Pot"). The agent plays a restaurant server handling customer interactions over phone or in person — taking reservations, resolving complaints, checking allergy safety, applying discounts, and coordinating with the kitchen, host, and manager.

Throughout this repo we refer to the project as **Hospitality Env** (short name) and **`hospitality_env`** (code identifier).

Built for the **OpenEnv Challenge** (Meta PyTorch × Hugging Face × Unsloth, 2026).

---

## Why this environment

Most existing agent benchmarks are either narrow tool-use tests (single-turn API calls) or generic customer-support simulations that treat every ticket the same. Front-line hospitality is different:

- Decisions are **multi-constraint** — a single allergy complaint triggers safety checks, authority limits on comps, incident logging, and de-escalation language all at once.
- The same surface request (*"I want a discount"*) has **very different correct responses** depending on membership status, shift timing, manager availability, and prior incident history.
- Tool calls and dialogue are **interleaved** — the agent must both talk to the customer and act on backend systems in the same turn.

**The content, data, and business processes in this environment are all designed from years of real front-line experience at a leading benchmark hospitality chain.** The policy document, tool taxonomy, task distribution, and reward rubric reflect how a real server actually thinks through these situations — not a synthetic "what would ChatGPT say" approximation.

This gives the environment a grounded, domain-expert signal that we believe is rare in public RL benchmarks.

---

## What's inside

```
hospitality_env/
├── models.py                        # Action / Observation schemas
├── client.py                        # Async HTTP client for the env
├── server/
│   ├── hospitality_env_environment.py   # Core env logic (reset / step / close)
│   ├── tools.py                          # 20+ restaurant tools
│   ├── tasks.py                          # 116 task scenarios
│   └── data/
│       ├── policy.md                     # 9 scenario playbooks
│       ├── menu.json
│       ├── customers.json
│       └── ...
```

- **116 tasks** covering 9 scenario families (allergy, incident, slow service, reservation, discount, comp, membership, lookups, host/seating).
- **20+ tools** with realistic authority limits (e.g. servers can comp a drink but not a whole bill; manager override required above threshold).
- **4-tier weighted reward**: safety (0.5) / authority (0.4) / procedural (0.25) / service (0.15).
- **Dual-standard close logic**: tasks end when either the expected tools are called *or* an 8-turn fallback triggers with a substantive reply + at least one tool call. This was tuned from baseline diagnostics (see `scratch/study_notes/` in development — not shipped publicly).

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/<your-username>/openenv-hospitality.git
cd openenv-hospitality
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Run the environment server

```bash
python -m hospitality_env.server
# listens on http://127.0.0.1:8000
```

### 3. Hello-world test

```bash
python hello_world_test.py
```

This connects an async client, calls `reset` and `step` a few times, and prints the observations.

### 4. Run a baseline eval

Evaluate a model across all 116 tasks:

```bash
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY, etc.
python baseline_eval.py \
    --model claude-sonnet-4-5 \
    --num-tasks 116 \
    --out-dir eval_results/
```

Results are written as JSON with per-task reward, turn count, tool calls, and conversation trace.

### 5. Compare runs

```bash
python compare_baselines.py eval_results/baseline_*.json
```

Prints mean reward, mean turns, max-turn hit rate, and per-scenario breakdown.

---

## Training

A Colab-ready GRPO training notebook is provided at `notebooks/train_qwen_grpo.ipynb` (Qwen2.5-1.5B + Unsloth 4-bit LoRA, r=16 α=32). It uses the same env server over HTTP, so training and eval share the exact same task distribution.

```bash
jupyter notebook notebooks/train_qwen_grpo.ipynb
```

---

## Baseline results

Evaluated with Claude Sonnet 4.5 on all 116 tasks:

| Version | Mean reward | Mean turns | Max-turn hits |
|---------|-------------|------------|---------------|
| v1 (initial) | 0.61 | 9.6 | 35 / 116 |
| v3 (close-logic fix) | 0.71 | 6.6 | 4 / 116 |

The v3 drop in mean turns (9.6 → 6.6) came from aligning the task-termination boundary with the reward boundary. See the project writeup for the full diagnosis.

---

## Citation

If you use this environment, please cite:

```
@misc{wang2026hospitality,
  title  = {Hospitality RL Environment: A Hot Pot Restaurant Simulation},
  author = {Binlei Wang},
  year   = {2026},
  howpublished = {OpenEnv Challenge submission}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
