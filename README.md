# Hospitality RL Environment: A Hot Pot Restaurant Simulation

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment that simulates a full-service hot pot restaurant ("Berkeley Hot Pot"). The agent plays a restaurant server handling customer interactions over phone or in person — taking reservations, resolving complaints, checking allergy safety, applying discounts, and coordinating with the kitchen, host, and manager.

Throughout this repo we refer to the project as **Hospitality Env** (short name) and **`hospitality_env`** (code identifier).

Built for the **OpenEnv Challenge** (Meta PyTorch × Hugging Face × Unsloth, 2026).

> **TL;DR** — A 116-task, 33-tool, multi-turn, safety-critical service-industry environment built from real hot-pot-restaurant operations. I ran the 2026 post-training stack against it end-to-end: **SFT** on three model families (Qwen, Llama, Gemma) at two capacity tiers (7B, 14B), and **GRPO** at two scales (1.5B on T4, 7B on A100 with SFT warmup). All six SFT configs landed inside a **0.056-wide reward band** (+0.068 to +0.124); Claude Sonnet 4.5 alone escaped at **+0.314**. Driving the best SFT recipe's training loss **12.9× lower** moved stratified eval **+0.003**. Both GRPO runs collapsed with `reward_std = 0` within 10–30 steps — no intra-group advantage signal at either scale.
>
> The null is what I'm submitting. SFT and GRPO each hit the same wall from opposite sides: hospitality is a *vertical-composite* domain, and a reward surface at 3.6 tasks/category is too thin for either method to find signal. The path V2 is taking is ImageNet-style per-class density — 10–50× the current coverage per sub-vertical, combined with step-wise reward shaping — not more recipe tuning on the same task inventory.

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

Three recipe iterations were trained. Only **v1** and **v3** produce comparable numbers on the final stratified held-out eval; v2 was retired when the stratified split revealed train/eval task-ID leakage in the earlier eval, so its number is not apples-to-apples:

| Version | Role | Epochs | Steps | Train loss (final) | Stratified eval | Notes |
|---|---|---|---|---|---|---|
| **v1** | Conservative release (shipped) | 3 | 18 | 0.7559 | +0.101 | Per-task rewards bit-identical to base |
| v2 | Aggressive (50 traj) — **retired** | 5 | — | — | *not comparable* | Scored high on an early eval split that had train/eval task-ID overlap; retired once the stratified split exposed the leakage |
| **v3** | Overfit stress test (held, not shipped) | 6 | 30 | **0.0587** ⚠️ | +0.104 | 12.9× loss reduction vs v1; held-out eval moved +0.003 |

**Released adapter:** [`binleiwang/qwen2.5-7b-hospitality-sft`](https://huggingface.co/binleiwang/qwen2.5-7b-hospitality-sft) — v1 (conservative recipe). v3 is held locally as the stress-test reference; v2 is not published.

### GRPO attempts — collapse at both scales

Before SFT became the canonical V1 release, I ran two GRPO attempts along the DeepSeek-R1 SFT-warmup-then-GRPO template. Both collapsed:

| Attempt | Stack | Outcome |
|---|---|---|
| 1 | Qwen 2.5 1.5B + LoRA r=16, pure GRPO, T4 (Colab) | `reward_std = 0` from step 1; 100 steps of zero advantage |
| 2 | Qwen 2.5 7B + LoRA r=16, 1-epoch SFT warmup → GRPO, A100 (Colab PAYG) | `reward_std = 0` by ~step 30; β·KL was the only remaining gradient, which pulled the SFT adapter back toward base — GRPO was actively unlearning the warmup |

GRPO computes advantage as `A_i = (reward_i − group_mean) / group_std` over K rollouts per group. When all K rollouts in a group land on the same terminal reward, the denominator is zero and no policy gradient exists. The HF Trainer HTML progress bar hides this: `loss` still prints small numbers because `loss = policy_ratio × advantage + β·KL`, and `β·KL ≈ 0` in the first steps. To see the failure you have to read `reward_std`, `grad_norm`, and `kl` directly from `trainer.state.log_history`.

This is the same problem the SFT null is pointing at, surfaced differently. A reward surface with 3.6 tasks/category and a terminal-only reward doesn't produce enough rollout-level variance for group-relative advantage to exist either. Archived scripts: `notebooks/_archive/train_qwen_grpo.py`, `train_qwen7b_sft_grpo.py`, `train_qwen7b_sft_grpo_v2.py`. Full diagnostic trail: `scratch/study_notes/03_dev_notes.md` §§负一, 负零点五, 负零点一.

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

Six open-weight configs, three different failure modes, one 0.056-wide band roughly 3× below Claude:

- **`frozen_loop`** (Qwen 7B family) — agent emits "I'd be happy to help with that" on repeat and never calls a tool.
- **`tool_spam`** (Qwen 14B, Llama 8B) — many tool calls, frequently the wrong ones, near-zero reward.
- **`chat_only`** (Gemma 9B) — coherent conversation, zero tool calls. Gemma still ties Qwen 14B at +0.068 via the opposite strategy from tool-spam.

### The optimization-vs-reward decoupling

v3 drove training loss **12× lower** than v1 (0.7559 → 0.0587 — near-complete memorization of the 40-trajectory SFT set). The stratified eval moved **+0.003**.

> The optimization landscape and the reward landscape are on different manifolds. Driving cross-entropy on imitation trajectories is not the signal the environment rewards.

### Where open models beat Claude

One category — `server_food_safety` — is the exception across the null band:

| Category | Gemma 9B base | Claude | Δ |
|---|---|---|---|
| `server_food_safety` | **+0.100** | −0.100 | **+0.200** |

Gemma wins this category by doing nothing. It never calls a tool, so it never triggers the policy-violation penalty Claude incurs when it *tries* to handle the allergy case. That's evidence the reward surface rewards inaction on policy-sensitive categories — one of the 8 structural limitations listed below.

Raw per-task outputs for all 7 model runs: `evals/eval_heldout_*.json`. Claude baselines: `eval_results/baseline_claude-sonnet-4-5_*.json`.

---

## Core contribution: the compositional density problem

The result above isn't a failed training run. SFT's pipeline works end-to-end, loss decreases monotonically, LoRA applies cleanly, the adapter loads and runs against the env. It's the *outcome* of that working pipeline that matters — a 0.056 band, a 12.9× loss drop that moved eval +0.003, and two GRPO collapses on top — and that outcome is what I think the submission is actually about.

### Hospitality is a composite of sub-verticals

Hospitality looks like a single vertical from the outside, but inside the restaurant it's 11 distinct sub-verticals:

- Host-side — phone, reservations, inquiries, walk-in seating
- Floor service — order management, out-of-stock coordination, multi-party billing, celebration
- Safety — allergen handling, incidents, food-poisoning claims, choking protocols
- Promotion — discounts, loyalty, vouchers, chain-wide promo inconsistencies
- Escalation — authority limits, manager override
- ...

Each sub-vertical has its own within-class invariance. "Allergen scenario in party X with member profile Y" and "allergen scenario in party A with member profile B" are the same *kind* of problem with structurally analogous solutions — but only if the agent has seen enough examples of that kind to extract the invariance. That's depth within the class, not breadth across classes.

### The ImageNet comparison

ImageNet didn't work because it had 1,000 classes. It worked because each class had ~1,300 images — enough per-class density for models to learn class-invariant features. V1 has 3.6 tasks per category.

The cross-family null (Qwen / Llama / Gemma clustering at [+0.068, +0.124]) and the v3 result (12.9× training-loss reduction → +0.003 eval) are both what "below the per-class-density threshold" looks like from the SFT side. GRPO at 1.5B and 7B shows the same thing from the RL side: when per-category coverage is too thin, K rollouts in a group converge on the same terminal reward, `reward_std → 0`, and the advantage denominator disappears before policy gradient can form. Two methods, two failure signatures, same root — the reward surface doesn't separate behaviors at this data density.

No recipe tuning, model scale, or architecture change crosses that threshold. Only data scale does.

### What V1 is

- Not a failed training run — the pipeline works.
- Not a benchmark — you can't rank models on it usefully, because the reward surface saturates on inaction (Gemma `server_food_safety` +0.100 vs Claude −0.100).
- A controlled experiment on whether a hospitality-domain RL environment can produce learning at benchmark-scale task coverage (116 tasks, 11 categories, ~3.6/category). Answer: no.

### What V2 changes

V2 isn't a bug-fix of V1. It re-specifies the two things V1 pinned down as bottlenecks, in priority order:

- **Data scale** (P0) — programmatic task generation pushing per-category coverage 10–50×, approaching ImageNet-like density per sub-vertical.
- **Reward surface** (P0–P1) — step-wise shaping on verified tool calls, explicit inaction penalties on policy-sensitive categories, trace-based assertion verification.
- **Eval hygiene** (P1) — N=3 sampled rollouts with confidence intervals, per-category variance reported.
- **Structural lifts** (P2–P3) — multi-agent floor parallelism (so the kitchen-lag pattern actually runs in parallel), shift-level memory, adversarial customer-simulator updates.

The high-level axes above are the narrative version. The **fully itemized roadmap — 8 structural limitations mapped to V2 fixes, priority-ordered P0 → P3 — is in the [V2 roadmap section](#v2-roadmap-8-limitations--fixes-p0p3) below.** Read that one before judging whether V2 has a coherent plan.

The generalization is the part I care about most: vertical-composite domains — hospitality, healthcare, legal, customer service — aren't going to yield to single-thread SFT or RL over benchmark-grade task inventories. They need per-class density of the kind ImageNet had for vision. V1 is the reference no-signal baseline against which V2's interventions will be measured. The released adapter, the 6 stratified eval JSONs, and the 4 Claude baseline runs are what a future comparison uses.

---

## V2 roadmap — 8 limitations → fixes (P0–P3)

This is the canonical V2 plan. As much of this submission is about what the environment can't train as what it enables — so this section is the single most load-bearing thing in the whole roadmap.

### Two independent, additive axes

V2 is not a single problem. The dev-notes diagnosis splits into two independent dimensions:

- **Data-scale axis** — V1 has 3.6 tasks/category vs ImageNet's ~1,300 images/class. Addressed by the 10–50× programmatic task generation mentioned in [What V2 changes](#what-v2-changes) above.
- **Schema-design axis** — 8 structural limitations in the env's policy / reward / simulator / metric layers, enumerated below.

These are **additive, not substitutable**. Even at 100 tasks/category, if the reward surface is still terminal-only (P0-2), SFT still can't learn multi-step orchestration and GRPO still can't find intra-group variance. Even at perfect reward shaping, if per-category coverage stays at 3.6, within-class invariance doesn't emerge. **V2 must fix both axes.** The data axis is the size of the training signal; the schema axis is its shape. The table below is the schema axis.

### The 8 schema-level limitations, by priority

| Priority | Limitation | V2 fix |
|---|---|---|
| **P0-1** (policy) | Policy is natural-language prose pasted into the system prompt; no 1:1 mapping between policy clauses, tools, and rubric assertions. Agents guess tool names from semantic cues and fail. | Structured policy records with `policy_id ↔ required_tools ↔ rubric_binding`. Policy lookup becomes an explicit tool call, not prose interpretation. |
| **P0-2** (reward) | Terminal-only reward — an 8-turn task returns one scalar at the end. **This is the specific mechanism behind our GRPO collapse:** rollout-internal reward variance is zero, so group advantage `(r_i − mean)/std` has a zero denominator. Sparse-gradient problem also blocks SFT credit assignment. | `process_assertions` per turn (+0.1 each verified sub-action) layered on top of `terminal_assertions`. Per-turn reward gradient, non-degenerate group advantage. |
| **P0-3** (reward) | Inaction is Pareto-optimal. Current shape: `correct → +x`, `wrong → −1.4`, `no action → 0`. Gemma's chat-only win on `server_food_safety` (+0.000 vs Claude's −0.100) is this shape exploited, not a bug in Gemma. | Per-category `inaction_floor: −0.3` when all process + terminal assertions fail **and** tool_calls == 0. Keeps a buffer above tool-spam's penalty but stops rewarding evasion. |
| **P1-4** (policy) | Ritual "look up the policy" assertions — but the policy is already in the system prompt. The test is "did the agent perform the ceremony," not "does the agent know." Top shared failure mode for Qwen 7B / Llama / Gemma. | Either (a) move policy out of system prompt, expose only via a `query_policy()` tool (preferred — matches real-restaurant SOP where staff actually consults the handbook), or (b) drop the ritual assertions entirely. |
| **P1-5** (simulator) | User simulator is one-way scripted: it has an *end-conversation* condition but no mid-trajectory correction. Agent goes off-rails → user waits silently → 8 turns → fail. Real customers prompt the correction; our simulator doesn't. | Add `nudge_policy` to the user simulator: when the agent has been silent or off-topic for 2 turns, inject "I still need you to {missing_action}" with some probability. Turns a cold test into a scaffolded env, makes multi-step learnable. |
| **P1-6** (fairness) | Rollout script injects an env-external `[SYSTEM NUDGE]` after 2 tool-less turns. Biases the benchmark toward tool-active families (Llama, Qwen 14B), penalizes conservative families (Gemma, Qwen 7B). Confound on cross-family comparison — what we report as "Gemma vs Qwen" is partly "Gemma vs Qwen *after a nudge layer*." | Either remove the nudge (expose natural policy differences), or make it a per-task flag `allow_system_nudge: bool` so both "proactive" and "restrained" task types are benchmarked separately. |
| **P2-7** (taxonomy) | `server_misc` (33 / 116 tasks) is "I don't know where to put it" — violates the `category = shared policy + tool + reward shape` invariant; breaks stratified sampling because misc has the largest within-category distribution shift. | Retire `server_misc`. Reclassify by downstream SOP: `accessibility_policy`, `lost_and_found`, `seating_preference`, etc. Each new category must share `relevant_policies`. |
| **P2-8** (metric) | No human baseline. Claude's +0.314 is assumed to be the ceiling, but Claude is a proxy, not a domain expert. We don't know which tasks are genuinely hard, which have multiple valid solutions, or whether +0.314 is actually above or below an experienced server's score. | Per-task `human_baseline`: `{server_experienced: score, server_novice: score, time_median_seconds: int}` collected from n=3 real servers. "Deploy-worthy" becomes "matches experienced-server score," not "beats Claude." |

Full derivation of each limitation with the specific V1 rollout evidence is in `scratch/study_notes/03_dev_notes.md` §负零点陆. Accompanying blog writeup: [the compositional density problem](https://binleiwang-hospitality-null-result-blog.static.hf.space/).

The null isn't about effort. I ran 6 SFT evaluations across 3 families × 2 capacity tiers, drove training loss 12.9× lower on the best recipe, and verified the same reward-surface problem from the GRPO side at two scales. The signal the environment needs isn't there in 40 imitation trajectories against a terminal-only reward over 11 thinly-sampled categories. V2 addresses the limitations above in priority order.

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
