---
title: Neural Paged Attention
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "4.36.1"
app_file: app.py
pinned: false
---

# Neural PagedAttention

> A reinforcement-learning environment for training agents to manage the **GPU KV Cache** in LLM inference servers.

Neural PagedAttention is a deterministic, high-fidelity simulator of [vLLM](https://github.com/vllm-project/vllm)-style PagedAttention memory hardware. It exposes the cache eviction / swapping / admission decisions as a [Gymnasium](https://gymnasium.farama.org/)-compatible RL environment, and ships with a suite of baseline and learned agents (LRU, Random, Q-Learning, DQN, LLM, PPO) so static and dynamic policies can be benchmarked head-to-head on identical workloads.

A full design write-up lives in [`blog.md`](./blog.md).

![Architecture diagram of the KV cache eviction environment](assets/kv_cache_system_architecture.png)

---

## Why this exists

Modern LLM servers are bottlenecked by GPU memory. Static heuristics like LRU/FIFO are simple and well-understood, but they cannot anticipate future requests, distinguish VIP from Free users, or react to bimodal traffic. Existing research into learned eviction policies is fragmented — every paper builds its own simulator and metrics.

This project provides a **shared, reproducible environment** where:

- Static baselines (LRU, FIFO, size-weighted) and learned policies (DQN, PPO, LLM-as-policy) run on identical scenarios.
- Traffic patterns include short chatters, power users, returning users, and viral spikes.
- Episodes are scored on a composite of throughput, latency efficiency, and cache residency.

It is built as an early contribution to **OpenEnv**, a framework for standardizing RL environments in LLM infrastructure.

---

## Features

- **Deterministic simulator** of PagedAttention with GPU + CPU swap tiers.
- **20-D observation space** capturing memory pressure, queue depth, idle-cache statistics, and SLA wait-time pressure.
- **18 discrete actions** spanning eviction, swapping, admission, rejection, preemption, and garbage collection.
- **Three difficulty tiers** — `easy`, `medium`, `hard` — with different traffic models and SLAs.
- **Six bundled agents** plug-and-play behind a single API:
  - `lru` — Least Recently Used heuristic
  - `random` — Random baseline
  - `qlearning` — Tabular Q-learning
  - `neural` — Deep Q-Network (PyTorch)
  - `llm` — Qwen2.5-3B-Instruct as a zero-shot policy
  - `ppo` — Unsloth-fine-tuned Qwen with REINFORCE/PPO + LoRA via MLX
- **FastAPI HTTP server** (`/api/simulate`, `/api/agents`, `/api/settings`) for headless batch runs and dashboards.
- **Next.js dashboard** under `dashboard/` for visualizing tick-level telemetry.
- **Hugging Face Spaces deploy** out of the box (this repo is a Docker SDK Space).

---

## Repository layout

```
.
├── app.py                  # FastAPI entrypoint (served on port 7860)
├── inference.py            # Standalone inference / smoke test
├── models.py               # Pydantic schemas for observations & actions
├── train_dqn.py            # DQN training loop
├── run_lru_test.py         # Quick LRU sanity check
├── run_qagent_test.py      # Quick tabular Q-learning sanity check
├── agents/                 # Agent implementations
│   ├── LRUAgent/
│   ├── RandomAgent/
│   ├── QLearningAgent/
│   ├── NeuralAgent/        # DQN
│   ├── LLMAgent/           # Qwen2.5 zero-shot
│   └── UnslothPPOAgent/    # PPO + LoRA via MLX
├── server/                 # Core environment
│   ├── environment.py      # KVCacheEnvironment (reset / step / state)
│   ├── env_components/     # Constants, traffic generator, scoring
│   └── session_manager.py  # Multi-session support
├── hf_train_space/         # Hugging Face Space for training
├── hf_ppo_space/           # Hugging Face Space for PPO inference
├── dashboard/              # Next.js telemetry dashboard
├── colab/                  # Colab training notebooks
├── assets/                 # Diagrams used in blog.md
├── blog.md                 # Project write-up
├── openapi.yaml            # OpenAPI spec for the HTTP API
├── Dockerfile              # Hugging Face Space build
├── requirements.txt
└── pyproject.toml
```

---

## Environment specification

### Observation space

20-dimensional `float32` array, normalized to `[0.0, 1.0]` (the `memory_pressure_trend` field is in `[-1.0, 1.0]`).

| #  | Field                     | Description                                                  |
|----|---------------------------|--------------------------------------------------------------|
| 1  | `gpu_utilization_pct`     | Fraction of GPU blocks in use                                |
| 2  | `cpu_utilization_pct`     | Fraction of CPU blocks in use                                |
| 3  | `memory_pressure_trend`   | GPU utilization slope over last 5 ticks (negative = falling) |
| 4  | `free_queue_pressure`     | Free queue length / max (100)                                |
| 5  | `vip_queue_pressure`      | VIP queue length / max (50)                                  |
| 6  | `free_max_wait_time_pct`  | Oldest Free request wait / SLA threshold                     |
| 7  | `vip_max_wait_time_pct`   | Oldest VIP request wait / SLA threshold                      |
| 8  | `yield_preempt_active`    | Largest active request size / GPU capacity                   |
| 9  | `free_size_max`           | Largest idle Free GPU cache / GPU capacity                   |
| 10 | `free_size_mean`          | Mean idle Free GPU cache size                                |
| 11 | `free_size_std_dev`       | Std dev of idle Free GPU cache sizes                         |
| 12 | `vip_size_max`            | Largest idle VIP GPU cache / GPU capacity                    |
| 13 | `vip_size_mean`           | Mean idle VIP GPU cache size                                 |
| 14 | `vip_size_std_dev`        | Std dev of idle VIP GPU cache sizes                          |
| 15 | `free_age_max`            | Oldest idle Free GPU cache age (normalized to 500 ticks)     |
| 16 | `free_age_mean`           | Mean idle Free GPU cache age                                 |
| 17 | `free_age_std_dev`        | Std dev of idle Free GPU cache ages                          |
| 18 | `vip_age_max`             | Oldest idle VIP GPU cache age                                |
| 19 | `vip_age_mean`            | Mean idle VIP GPU cache age                                  |
| 20 | `vip_age_std_dev`         | Std dev of idle VIP GPU cache ages                           |

### Action space

`Discrete(18)`:

| ID | Action                                                                |
|----|-----------------------------------------------------------------------|
| 0  | Evict Largest Free cache (idle GPU only)                              |
| 1  | Evict Largest VIP cache (idle GPU only)                               |
| 2  | Evict Oldest Free cache (idle GPU only)                               |
| 3  | Evict Oldest VIP cache (idle GPU only)                                |
| 4  | Swap Largest Free cache GPU→CPU                                       |
| 5  | Swap Largest VIP cache GPU→CPU                                        |
| 6  | Swap Oldest Free cache GPU→CPU                                        |
| 7  | Swap Oldest VIP cache GPU→CPU                                         |
| 8  | Admit next Free user from queue                                       |
| 9  | Admit next VIP user from queue                                        |
| 10 | Reject next Free user (penalty if GPU has space)                      |
| 11 | Reject next VIP user (penalty if GPU has space)                       |
| 12 | Preempt & Shred Largest Active Free request                           |
| 13 | Preempt & Shred Largest Active VIP request                            |
| 14 | Preempt & Swap Largest Active Free → CPU                              |
| 15 | Preempt & Swap Largest Active VIP → CPU                               |
| 16 | Garbage Collect (delete idle Free CPU caches > 200 ticks)             |
| 17 | Do Nothing                                                            |

### Tasks

| Task   | Traffic        | SLA Free | SLA VIP | Max Ticks |
|--------|----------------|----------|---------|-----------|
| easy   | 1 req / tick   | None     | None    | 2,000     |
| medium | Day / night    | 100      | 50      | 5,000     |
| hard   | Viral spikes   | 50       | 25      | 10,000    |

A tick corresponds to one token generated across all active requests. The default GPU has 10,000 blocks (≈160k tokens) and the CPU swap tier has 50,000 blocks (≈800k tokens). Traffic mix defaults to ~80% chatters, ~20% power users, with ~30% returning users.

### Scoring

Each episode produces a composite score over:

- **Throughput** — fraction of arriving requests successfully completed.
- **Latency efficiency** — generation time vs. queue-wait time per request.
- **Cache residency** — for returning users, whether their historical cache survived.

Throughput is the dominant term; cache residency acts as a tie-breaker.

---

## Quickstart

### Prerequisites

- Python **≥ 3.10**
- (Optional) Docker, for the Hugging Face Space build
- (Optional) `HF_TOKEN` if you want to pull gated models

### Local install

```bash
git clone https://github.com/<your-org>/NeuralPagedAttention.git
cd NeuralPagedAttention

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the API server

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

The server is now available at `http://localhost:7860` with interactive Swagger docs at `http://localhost:7860/docs`.

### Run with Docker

```bash
docker build -t neural-paged-attention .
docker run -p 7860:7860 neural-paged-attention
```

### Smoke-test an agent

```bash
python run_lru_test.py
python run_qagent_test.py
```

### Train the DQN agent

```bash
python train_dqn.py
```

### Run inference with the LLM agent

```bash
export HF_TOKEN=your_token
export MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python inference.py
```

---

## HTTP API

Full schema lives in [`openapi.yaml`](./openapi.yaml). The most useful endpoints:

### `GET /api/agents`
List the agents available to the simulator.

### `GET /api/settings` · `POST /api/settings`
Inspect or override environment constants (`GPU_TOTAL_BLOCKS`, `CPU_TOTAL_BLOCKS`, `TOKENS_PER_BLOCK`, per-task `max_ticks`, `TRAFFIC_SEED`).

### `POST /api/simulate`
Run a full headless simulation and return tick- and session-level telemetry.

```bash
curl -X POST http://localhost:7860/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"agent": "neural", "task": "hard"}'
```

Response:

```json
{
  "status": "success",
  "agent": "neural",
  "task": "hard",
  "session_logs": [ ... ],
  "tick_logs":   [ ... ]
}
```

Legacy single-env endpoints (`/reset`, `/step`, `/state`) remain available but are deprecated; prefer the session-scoped variants exposed via `server/session_manager.py`.

---

## Baseline scores

The table below reports the **environment final score** (a composite of throughput, latency efficiency, and cache residency, clamped to `(0.001, 0.999)`). Random / LRU / Q-Learning / DQN were measured locally with the default environment constants. LLM and PPO scores were fetched from the same backend the dashboard uses (`https://suryanshchattree-neural-paged-attention-env.hf.space/api/simulate`), where their tick budget is capped to keep request latency manageable on a CPU-only Space.

| Agent              | easy  | medium | hard  | Notes                                              |
|--------------------|-------|--------|-------|----------------------------------------------------|
| Random             | 0.887 | 0.001  | 0.001 | Crashes within ~15 ticks on medium / hard          |
| **LRU**            | **0.977** | **0.253** | 0.223 | Best on `easy` and `medium`; only agent that survives all of `easy` |
| Tabular Q-Learning | 0.435 | 0.172  | 0.148 | Cold-start, no warm Q-table — bottom of the pack    |
| DQN (Neural)       | 0.898 | 0.001  | 0.353 | Strong on `easy`, crashes early on `medium`         |
| LLM (Qwen2.5-1.5B) | 0.267 | 0.309  | 0.333 | Hosted run; ~50-tick budget. Parse-failure rate **0%** |
| PPO (Unsloth + LoRA) | 0.334 | 0.253 | **0.352** | Hosted run; survives furthest on `medium` (138 ticks) before crash |

Cumulative-score traces from the live dashboard make the gap between difficulty tiers easy to read: every agent finishes `easy` in positive territory, while every agent is in free fall by the end of `hard`.

![Cumulative score over time on the easy task — LRU, PPO, NEURAL and LLM all complete the run, with LRU well clear of the rest.](assets/Reward_easy.png)

![Cumulative score over time on the hard task — every agent trends sharply negative as the queues overwhelm GPU capacity.](assets/Reward_hard.png)

Raw per-run telemetry (reward totals, ticks run, arrival/completion counts, crash flags, LLM parse failures) is saved to [`results/benchmarks.json`](./results/benchmarks.json) and the full LLM/PPO tick-level traces are in [`results/remote_runs/`](./results/remote_runs/).

A few takeaways:

- **`easy` rewards a near-do-nothing policy more than people expect.** Even Random scores `0.887` because traffic stays under capacity. The environment doesn't really start *testing* a policy until `medium`.
- **`medium` is the difficulty cliff.** Random and DQN both trip the crash penalty within the first 20 ticks, collapsing to the `0.001` floor. LRU's `0.253` is the strongest passing score.
- **LLM and PPO get higher scores than Random / DQN on the harder tiers but on much shorter horizons.** Both crash within ~25–30 ticks on `medium`/`hard` on the hosted backend. The composite score is biased upward by the small denominator (few requests arrived before crash). PPO on `medium` is the exception: it sustains 138 ticks and 12 completed requests, the best LLM-class number on that tier.
- **The classical baseline is harder to beat than people assume.** LRU's `0.977` on `easy` and `0.253` on `medium` are the bars a learned policy actually has to clear. None of the learned agents shipped here clear them on `medium` end-to-end.

To regenerate locally:

```bash
python scripts/run_benchmarks.py                     # random / lru / qlearning / neural
python scripts/fetch_remote_scores.py                # llm / ppo via the live HF Space
# or, with GPU + cached Qwen2.5:
python scripts/run_benchmarks.py --include-llm --include-ppo
```

Or run an individual agent against the live API:

```bash
curl -s -X POST https://suryanshchattree-neural-paged-attention-env.hf.space/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"agent": "ppo", "task": "medium"}' \
  | jq '.session_logs[].final_score'
```

---

## Live dashboard

A web dashboard for running and inspecting these simulations is hosted at:

**[https://neural-paged-attention-dashboard-7q.vercel.app/](https://neural-paged-attention-dashboard-7q.vercel.app/)**

It is a Vite + React + Recharts client that talks to the public Hugging Face Space backend at `https://suryanshchattree-neural-paged-attention-env.hf.space/api`.

### What the dashboard shows

- **Configuration panel** — override the global environment constants (deterministic traffic seed, GPU and CPU total blocks, tokens per block, max ticks per difficulty), then trigger a *batch* run with the **START BATCH SIMULATION** button. Leaving fields blank uses the backend defaults.
- **Multi-agent comparison panel** — `MULTI-AGENT COMPARISON (CUMULATIVE SCORE)` plots the running cumulative score of each agent on the same axis so you can read off relative performance over time, not just at episode end.
- **Agent focus panel** — `AGENT FOCUS:` lets you drill into a single agent and shows live `Tick`, `Score`, `Total Reward`, current `Task`, `Action`, `GPU / CPU Util`, `Memory Trend`, and `Yield Preempt` indicator.
- **Queue panels** — separate `Free Requests` and `VIP Requests` charts surface the queue dynamics that drive the SLA-miss penalties.
- **Live log** — per-tick action / reward / tick stream (`log-action`, `log-reward`, `log-tick`) and a `Live Score` running total.
- **Status badges** — `COMPLETED` and `CRASHED` flags per session, matching the `crashed` field in the API's `session_logs`.

A completed `easy` run vs. a fully-crashed `hard` run, side by side on the dashboard's traffic / token / macro-stats panels:

![Easy run dashboard view — Free/VIP queues stay near zero, token processing tops out around 1.5k, and the per-agent macro stats panel shows COMPLETED status for LRU, NEURAL, LLM, and PPO.](assets/Session_easy.png)

![Hard run dashboard view — Free queue saturated at 100 and VIP queue saturated at 50, generated-token bursts above 45k/tick, all four agents in CRASHED status.](assets/Session_Hard.png)

### Insights you can read off the dashboard

1. **Crash localization.** The cumulative-score line for an agent flattens the moment its `crashed` flag flips. Random and DQN go flat almost immediately on `medium` / `hard`, while LRU keeps drifting and only hits the wall mid-episode.
2. **Memory-pressure inflection points.** The `Memory Trend` and `GPU Utilization` charts reveal that crashes follow a sharp, concentrated 5-tick rise in GPU utilization rather than slow drift — agents that don't pre-empt aggressively lose the window.
3. **Throughput vs. latency trade-off.** On the LRU run, you can see a tight queue (`VIP Requests` near zero) but periodic stalls in cumulative score during traffic spikes — that's the policy paying for swap latency to keep VIPs admitted.
4. **Policy fingerprints.** The action stream in the live log makes each agent's signature obvious: Random is uniform, LRU concentrates on `Swap Oldest` / `Evict Oldest` once GPU > 85%, the Q-learning agent over-issues `Reject` actions, and DQN's behavior changes character around the same memory-pressure thresholds the LRU heuristic uses.
5. **Reward shape ≠ score shape.** Cumulative reward can plunge into very negative territory (large SLA penalties) while final score remains > 0.2 — useful when comparing crashed-but-completed runs against early-terminated ones.

---

## Limitations

This is an early-stage simulator and intentionally simplified.

- No hardware-level timing effects, memory bandwidth, or PCIe transfer cost modeling.
- Traffic patterns are heuristic (compound sine + spikes), not replayed production traces.
- Not yet validated against a live vLLM deployment; simulator and real-world performance may diverge.

See the *"What it does not do"* section of [`blog.md`](./blog.md) for a longer discussion.

---

## Links

- 🌐 **Live dashboard:** https://neural-paged-attention-dashboard-7q.vercel.app/
- 🤗 **Hugging Face Space (backend):** https://suryanshchattree-neural-paged-attention-env.hf.space
- 📓 **Blog / design write-up:** [`blog.md`](./blog.md)
- 📜 **OpenAPI spec:** [`openapi.yaml`](./openapi.yaml)


---



