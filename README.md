# Hospitality RL Environment: A Hot Pot Restaurant Simulation

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment that simulates a full-service hot pot restaurant ("Berkeley Hot Pot"). The agent plays a restaurant server handling customer interactions over phone or in person — taking reservations, resolving complaints, checking allergy safety, applying discounts, and coordinating with the kitchen, host, and manager.

Throughout this repo we refer to the project as **Hospitality Env** (short name) and **`hospitality_env`** (code identifier).

Built for the **OpenEnv Challenge** (Meta PyTorch × Hugging Face × Unsloth, 2026).

> **TL;DR** — A 116-task, 33-tool, multi-turn, safety-critical service-industry environment built from real hot-pot-restaurant operations. We ran an honest SFT experiment across three model families (Qwen, Llama, Gemma) × two capacity tiers (7B, 14B). All six open-weight configurations clustered in a **0.056-wide band** (+0.068 to +0.124); Claude Sonnet 4.5 alone at **+0.314**. On the best SFT recipe, training loss dropped **12× lower** — stratified eval moved **+0.003**.
>
> **This null is the contribution, not the failure mode.** It empirically validates a specific thesis: hospitality is a *vertical-composite* domain — single-thread training on benchmark-scale task inventories (3.6 tasks/category) cannot close the gap to a frontier ceiling. **ImageNet-scale per-class density, not benchmark-scale breadth, is the path forward.** V2 re-specifies data scale and reward surface along the lines this V1 experiment identified.

---

## Why this environment

### The gap between today's voice AI and real service industry

Current voice-AI deployments in hospitality are confined to low-complexity transactional slices:

- **Drive-thru order taking** — SoundHound AI's Smart Ordering now powers [10,000+ restaurant locations](https://investors.soundhound.com/news-releases/)
- **Phone reservation bots** and IVR-style FAQ answering
- **Simple online chat** for booking confirmations and order tracking

The AI-in-restaurants market reflects this confinement: [USD 6.1B in 2024, projected to reach USD 48.3B by 2033 (CAGR 23.5%)](https://dataintelo.com/report/ai-in-restaurants-market). But the growth is overwhelmingly concentrated in these transactional slices. **Full-service floor operations — the bulk of the actual industry — remain untouched.**

### What today's deployments cannot do

Full-service hospitality requires capabilities transactional voice AI does not have:

- **Multi-constraint decisions** — a single allergy complaint simultaneously triggers safety protocol verification, authority limits on compensation, incident logging, and de-escalation language management.
- **Context-dependent correctness** — the same surface request (*"I want a discount"*) has very different correct responses depending on membership tier, shift timing, manager availability, and prior incident history. There is no one-size-fits-all answer.
- **Interleaved tool use and dialogue** — the agent must talk to the customer *and* act on backend systems in the same turn, under real-time pressure.
- **Edge-case density** — allergen cross-contamination, rush-hour kitchen backlog, chain-wide promotion inconsistency, authority escalation under time pressure: the categories where deployed AI most consistently fails.

### The embodied-AI dimension

The next frontier is embodied service robotics. Yet while conversational AI has rapidly matured, **restaurant serving robots have been stuck at the "carry-food-to-table" level for nearly a decade** — physical capability has far outpaced the domain reasoning required to deploy it usefully. A serving robot that can navigate a floor is useless without a brain that can decide *when* to approach a table, *how* to apologize for a delayed dish, or *whether* an item should be delivered to a guest who just flagged an allergy concern. **The bottleneck is domain cognition, not hardware.** Building that cognitive layer requires domain-faithful environments to train in. This repo is a first attempt.

### Why hot pot restaurants as the load-bearing vehicle

A high-end hot pot restaurant is an ideal concrete vehicle for the broader service-industry class because it naturally subsumes the full surface area of the problem:

- **Host-side**: phone reception, reservations, party-size inquiries, waitlist management — the same surface as today's deployed voice AI.
- **Floor service**: order management, out-of-stock coordination, multi-party billing, celebration handling — the multi-turn, emotionally-varied regime today's AI cannot handle.
- **High-stakes safety**: hot-soup spill liability, allergen cross-contamination (with a bubbling communal pot at every table), choking protocols — categories where errors are unforgiving.

### Domain grounding

The content, data, and business processes in this environment are designed from the author's years of front-line and management experience at an Asian hot pot chain widely regarded as the quality benchmark for Chinese hospitality service. The 460-line policy document, tool taxonomy, task distribution, and reward rubric reflect how a real server actually thinks through these situations — not a synthetic "what would ChatGPT say" approximation. This gives the environment a grounded, domain-expert signal that we believe is rare in public RL benchmarks.

---

## What's inside

```
hospitality_env/
├── models.py                       # Action / Observation schemas
├── client.py                       # Async HTTP client
├── openenv.yaml                    # OpenEnv config
├── pyproject.toml
└── server/
    ├── hospitality_env_environment.py   # reset / step / close
    ├── app.py                           # FastAPI server entry
    ├── Dockerfile                       # deployable image
    ├── data/
    │   ├── tasks.json                   # 116 task definitions
    │   ├── policy.md                    # Operational policies (~460 lines)
    │   ├── db.json                      # Restaurant state
    │   └── user_db.json                 # Customer database
    └── domain/
        ├── tools.py                     # 33 agent tools
        ├── user_tools.py                # 8 user-simulator tools
        ├── data_model.py                # Pydantic models
        └── utils.py
```

- **116 tasks across 11 categories**: host (`host_phone` 13, `host_seating` 6, `host_walkin` 1) · server (`server_food_safety` 11, `server_promotion` 16, `server_food_issue` 7, `server_billing` 6, `server_celebration` 4, `server_incident` 13, `server_special_policy` 6, `server_misc` 33).
- **33 agent tools + 8 user-simulator tools** with realistic authority limits (servers can comp small items but not whole bills; manager override required above threshold).
- **Deterministic evaluation** via ACTION (required tool calls) + ENV_ASSERTION (DB state checks), not LLM-as-judge.
- **Grounded in real operations** — policy, menu, table plan, staff hierarchy, and task scenarios all derived from an actual restaurant operation, extending the `hospitality` domain first contributed by this author to Sierra Research's [`tau2-bench`](https://github.com/sierra-research/tau2-bench) (2026, AgentBeats Special Mention).
- **Dual-standard close logic**: tasks end when either the expected tools are called *or* an 8-turn fallback triggers with a substantive reply + at least one tool call. This was tuned from baseline diagnostics during V1 development.

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
uvicorn hospitality_env.server.app:app --host 127.0.0.1 --port 8000
# listens on http://127.0.0.1:8000
```

### 3. Try a few tasks with Claude

```bash
python demo_3_tasks.py    # 3 representative tasks with Claude Sonnet 4.5
python demo_6_more.py     # 6 additional scenarios with trace output
```

These connect an async client, drive a Claude agent through several tasks end-to-end, and print the full trajectory — the simplest on-ramp to seeing the environment in action.

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

**`notebooks/train_lambda.ipynb`** — Qwen 2.5 7B Instruct + LoRA (r=16, α=32) + SFT on 40 high-reward Claude Sonnet 4.5 trajectories drawn from a stratified train split (40 train / 20 held-out, preserving per-category distribution). Runs on a single H100 in ~30 min.

```bash
jupyter notebook notebooks/train_lambda.ipynb
```

Three recipe iterations produced:

| Version | Epochs | Steps | Train loss (final) | Stratified eval | Notes |
|---|---|---|---|---|---|
| **v1** (conservative) | 3 | 18 | 0.7559 | +0.101 | Identical per-task rewards to base |
| v2 (aggressive, 50 traj) | 5 | — | — | *not run on stratified* | Overfit; inflated score on old ID-biased eval came from train/eval ID overlap |
| **v3** (pullback, 6 epoch) | 6 | 30 | **0.0587** ⚠️ | +0.104 | 12× loss reduction vs v1; eval moved +0.003 |

**Released adapter:** [`binleiwang/qwen2.5-7b-hospitality-sft`](https://huggingface.co/binleiwang/qwen2.5-7b-hospitality-sft) — v1 (conservative recipe).

Earlier GRPO attempts (1.5B-T4 and 7B-A100, both of which hit `reward_std = 0`) are preserved in `notebooks/_archive/` as diagnostic references for readers debugging similar failures on this stack. The pivot from RL to SFT-only was driven by the `reward_std = 0` failure mode, which itself is diagnostic of the reward-surface issues documented in §[What we learned](#what-we-learned).

---

## Baseline results — the triple-family null

All models below were evaluated on the **same 20-task stratified held-out set**, constructed to preserve the per-category distribution of the full 116-task benchmark (see `sft_data/stratified_manifest.json` for the split).

| Model | Stratified mean | Δ vs Claude | Mean turns | Failure mode |
|---|---|---|---|---|
| **Claude Sonnet 4.5** | **+0.314** | — (ceiling) | 6.15 | — |
| Qwen 2.5 7B (base) | +0.101 | −0.213 | 6.10 | `frozen_loop` |
| Qwen 2.5 7B + SFT v1 | +0.101 | −0.213 | 6.10 | `frozen_loop` (identical per-task to base) |
| Qwen 2.5 7B + SFT v3 | +0.104 | −0.210 | 6.25 | `frozen_loop` |
| Qwen 2.5 14B (base) | +0.068 | −0.246 | 6.80 | `tool_spam` |
| Llama 3.1 8B (base) | +0.124 | −0.190 | 5.90 | `tool_spam` |
| Gemma 2 9B (base) | +0.068 | −0.246 | 6.80 | `chat_only` |

**Three independent failure modes across three model families, all clustered in a 0.056-wide band roughly 3× below the Claude ceiling.**

- **`frozen_loop`** (Qwen 7B family) — agent emits "I'd be happy to help with that" repeatedly, never calls tools.
- **`tool_spam`** (Qwen 14B, Llama 8B) — agent calls many tools, frequently the wrong ones, netting near-zero reward.
- **`chat_only`** (Gemma 9B) — agent holds a coherent conversation but never calls a tool. Despite this, Gemma ties Qwen 14B's score (+0.068) — via the **opposite** strategy.

### The optimization-vs-reward decoupling

v3 drove training loss **12× lower** than v1 (0.7559 → 0.0587 — near-complete memorization of the 40-trajectory SFT set). The stratified eval moved **+0.003**.

> The optimization landscape and the reward landscape are on different manifolds. Driving cross-entropy on imitation trajectories is not the signal the environment rewards.

### Where open models beat Claude

One category — `server_food_safety` — is the exception across the null band:

| Category | Gemma 9B base | Claude | Δ |
|---|---|---|---|
| `server_food_safety` | **+0.100** | −0.100 | **+0.200** |

Gemma wins this category **by doing nothing**. It never calls a tool and so never triggers the policy-violation penalty Claude incurs when *attempting* to handle the allergy case. This is the cleanest single piece of evidence in the repo that **the reward surface rewards inaction on policy-sensitive categories** — one of the 8 structural limitations documented below.

Raw per-task outputs for all 7 model runs: `evals/eval_heldout_*.json`. Claude baselines: `eval_results/baseline_claude-sonnet-4-5_*.json`.

---

## Core contribution: the vertical-composite thesis

**The null result above is not a failed SFT run. It is a controlled experiment whose outcome matches a specific prediction about how service-industry environments must be built.**

### The thesis

Hospitality, despite appearing as a single vertical, is internally **a composite of sub-verticals**:

- Host-side phone / reservation / inquiry
- Floor service: order management, billing, celebration
- Safety: allergen handling, incidents, food-poisoning claims
- Promotion: discounts, loyalty, vouchers
- Escalation: authority limits, manager override
- ... (11 categories in this env)

Each sub-vertical requires its own **within-class invariance** — the agent must recognize that "allergen scenario in party X with member profile Y" and "allergen scenario in party A with member profile B" are the same *kind* of problem with structurally analogous solutions, despite different surface details. Learning that invariance requires **depth within the class**, not breadth across classes.

### The ImageNet analogy

ImageNet did not succeed because it had many classes — it succeeded because each class had ~1300 images, enough for models to learn class-invariant features. Our V1 has an average of **3.6 tasks per category**.

The cross-family null — Qwen / Llama / Gemma all clustering at [+0.068, +0.124], with v3 driving train loss 12× lower than v1 for a +0.003 eval movement — is the empirical signature of environments **below the data-scale threshold for invariance learning**. No amount of recipe tuning, model scale, or architecture change will cross that threshold. Only data scale will.

### What V1 is (and is not)

- **V1 is not a failed training run.** The SFT pipeline works end-to-end, losses decrease monotonically, LoRA applies cleanly, the adapter loads and runs on the env.
- **V1 is not a benchmark either** — you cannot rank models on it usefully, because the reward surface saturates on inaction (see Gemma `server_food_safety` +0.100 vs Claude −0.100).
- **V1 is a controlled experiment** that tests whether a hospitality-domain RL environment can produce learning at benchmark-scale task coverage (116 tasks across 11 categories, ~3.6/category). The answer is: no.

### What this means for V2 (and for the field)

V2 is not a bug-fix of V1. It is a **re-specification of scale and reward structure** along the lines V1's experiment identified:

- **Data scale** — programmatic task generation to push per-category coverage 10–50× (approaching ImageNet-like density per sub-vertical).
- **Reward surface** — step-wise shaping, inaction penalties, trace-based assertion verification (the 8 structural limitations enumerated in [What we learned](#what-we-learned)).

The implication generalizes: **any genuinely vertical-composite domain — hospitality, healthcare, legal, customer service — requires ImageNet-style per-class density, not benchmark-grade per-class sparseness, to train agents that generalize within class.** Single-thread SFT or RL on benchmark-scale task inventories will not close the gap to a Claude-level ceiling. This is what this repo, collectively, argues.

The V1 artifact set (this repo + the released adapter + the 6 stratified eval jsons + the 4 Claude baseline runs) is the **reference no-signal baseline** against which V2's data-scale and reward-surface interventions will be measured.

---

## What we learned

This submission is as much about **what the environment cannot train** as what it enables. Eight structural limitations were identified during V1 development and documented alongside V2 fixes:

| Priority | Limitation | V2 fix |
|---|---|---|
| **P0** | Policy is prose, not machine-checkable — agents must interpret 460 lines of natural language | Policy DSL + automated assertion generation |
| **P0** | Reward is terminal-only; partial tool progress is invisible | Per-step reward signal on each verified tool call |
| **P1** | Inaction is never penalized (see Gemma `server_food_safety` result) | Explicit "failure to act" negative terms on stateful categories |
| **P1** | Benchmark-scale data (3.6 tasks/category) — not env-scale (ImageNet: ~1300/class) | Programmatic task generation targeting 10–50× current per-category coverage |
| **P2** | Ritual assertions (hardcoded policy acknowledgments) inflate scores when agent parrots policy phrases | Trace-based assertion verification (did the agent *do* the thing, not just say it) |
| **P2** | Scripted user simulator lacks persona / emotional variability | LLM-judge user simulator with persona sampling |
| **P3** | Rollout-nudge bias — SFT data collection prompt subtly differs from eval prompt | Unified prompt template across collection + eval + training |
| **P3** | `server_misc` (33/116) is a taxonomy smell | Split into functional sub-categories |

Full analysis of each limitation with reproduction instructions is documented in the accompanying [blog writeup](https://huggingface.co/blog/TODO-blog-url).

**The null result is not a failure of effort.** We ran 6 independent model evaluations across 3 families × 2 capacity tiers, drove training loss 12× lower on the best SFT recipe, and verified the pattern across freeze / tool-spam / chat-only failure modes. The signal the environment needs simply does not exist in 40 imitation trajectories evaluated against a terminal-only reward over 11 thinly-sampled categories. **V2 addresses these root causes in the order listed above.**

---

## Connection to prior work

This environment extends the `hospitality` domain that the same author contributed to Sierra Research's [$\tau^2$-bench](https://github.com/sierra-research/tau2-bench) earlier in 2026 (AgentBeats Special Mention). The V0 on $\tau^2$ was **benchmark-grade** — deterministic evaluation over 116 static task definitions. This V1 on OpenEnv is the **env-grade extension** — training loop, reward surface, SFT pipeline, multi-family baseline diagnostics. The jump from benchmark to environment is where most of the structural limits surface, and this submission documents that transition longitudinally.

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
