# Neural PagedAttention: A Benchmarking Environment for KV Cache Eviction Policies

---

## Background: Where Memory Fits In

When a large language model generates a response, it maintains a running memory of every token it has processed so far. This memory structure is called the **KV Cache** (Key-Value Cache). The longer the conversation, the larger the cache — and GPU memory is finite.

[vLLM](https://github.com/vllm-project/vllm) addressed the allocation problem well. Its **PagedAttention** mechanism borrowed the idea of virtual memory paging from operating systems: rather than reserving a contiguous chunk of GPU memory per request upfront, it splits caches into fixed-size blocks and allocates them on demand. This eliminated a large source of waste.

What PagedAttention doesn't prescribe is: *when memory is full, what do you throw out?* That decision belongs to an **eviction policy**.

---

## Static Policies: The Established Standard

The overwhelming majority of production LLM servers today use **static eviction policies** — rules that are defined once and applied consistently. The most common are:

- **LRU (Least Recently Used):** Evict the cache that has gone untouched the longest.
- **FIFO (First In, First Out):** Evict the oldest cache in the system.
- **Size-weighted variants:** Prioritize evicting large caches to free the most space quickly.

These policies are deservedly dominant. They are simple to implement, deterministic, and well-studied. Critically, they perform close to optimal under many real workloads — and because their behavior is predictable, engineering teams can build capacity planning and SLA forecasts on top of them with confidence.

It is also worth noting that static policies are not as inflexible as they might first appear. With sufficient engineering, you can layer business logic on top: a tiered LRU that assigns higher priority to paying users, a rule that never evicts a cache belonging to an active session, or a size threshold below which caches are always retained. These extensions are mature and widely deployed.

---

## Where Static Policies Reach Their Ceiling

The core limitation of static policies is not that they lack sophistication — it is that *as humans, we can only reason so far ahead*. Any rule we write is a best guess about what the future looks like.

Consider a scenario: a server is operating at 90% GPU memory, and a sudden spike in traffic arrives. A static rule must decide what to evict using only the information visible at that moment. It cannot weigh the downstream cost of evicting one cache versus another — it can only apply its rule.

Over time, the kinds of workloads LLM servers handle have grown more complex: users with long, multi-session conversations, mixed traffic of very short and very long requests, and unpredictable bursts. As the gap between *what the rule assumes* and *what actually happens* widens, the cost of that gap grows.

![Bimodal distribution of LLM request lengths](assets/bimodal_request_distribution.svg)

This is the gap that dynamic policies try to close.

---

## The Case for Dynamic Eviction

The idea is straightforward: instead of a fixed rule, train a policy that *learns* from experience. Given a snapshot of the current system state — memory pressure, queue depth, who is waiting and for how long — a learned agent decides what to evict.

A handful of research papers have explored this direction, using reinforcement learning agents and generative planners. The results are genuinely interesting. But the research is fragmented: each paper builds its own simulator, defines its own metrics, and evaluates in isolation. There is currently no shared environment where static and dynamic policies can be compared on identical ground.

This is partly what motivated this project.

---

## What We Built (And What It Is Not)

This environment is an attempt to provide a common testing ground — a simulator where you can run a standard LRU policy and a trained RL agent through the same scenario and compare them on the same metrics.

It models a simplified GPU memory system with an overflow buffer (representing CPU swap space), a queue of incoming requests, and the eviction decisions that connect them. An agent observing the system state can choose to evict caches by size or age, swap them to the overflow buffer, accept or reject incoming requests, or in extreme cases, interrupt an active generation to reclaim memory — a costly but sometimes necessary action.

Training is structured in three stages of increasing difficulty: learning the basic mechanics, managing queues under time pressure, and protecting high-priority users during traffic spikes.

![Architecture diagram of the KV cache eviction environment](assets/kv_cache_system_architecture.png)

**What the environment does well:**
- Provides a reproducible, deterministic setup for comparison
- Supports both static baselines and trainable RL agents within the same interface
- Models a realistic mix of request types and arrival patterns
- Includes returning users whose cached memory may or may not still be present — a common real-world scenario

**What it does not do:**
- The simulation is significantly simplified. Real GPU memory management involves hardware-level timing effects, memory bandwidth constraints, and PCIe transfer costs that this environment does not model.
- Traffic patterns are generated from heuristics and intuition — a compound sine wave overlaid with random spikes — rather than from traces of actual production traffic. How well these patterns represent real workloads is an open question.
- The environment has not been validated against a live vLLM deployment. Simulator performance and real-world performance may diverge.

We are among the early contributors to OpenEnv, a new framework for standardizing RL environments in the LLM infrastructure space. This environment is a first step, not a finished answer.

---

## Evaluation

The environment scores a policy across three dimensions at the end of each episode:

- **Throughput:** What fraction of arriving requests were successfully completed?
- **Latency efficiency:** How much of each request's time was spent actually generating, versus waiting in queue?
- **Cache residency:** When a returning user arrived, was their historical cache still intact and usable?

The composite score is weighted toward throughput, with cache residency acting as a tie-breaker between policies that complete similar numbers of requests. This scoring can be adjusted — it reflects one reasonable set of priorities, not a universal truth.

### What the baselines actually look like

We ran every shipped agent on every task. Random, LRU, Q-Learning, and DQN were measured locally; LLM and PPO were measured on the same hosted backend the dashboard uses (`https://suryanshchattree-neural-paged-attention-env.hf.space/api/simulate`), where their per-request tick budget is capped to keep latency manageable. Final score is the composite above, clamped to `(0.001, 0.999)`:

| Agent              | easy  | medium | hard  |
|--------------------|-------|--------|-------|
| Random             | 0.887 | 0.001  | 0.001 |
| **LRU**            | **0.977** | **0.253** | 0.223 |
| Tabular Q-Learning | 0.435 | 0.172  | 0.148 |
| DQN (Neural)       | 0.898 | 0.001  | 0.353 |
| LLM (Qwen2.5-1.5B) | 0.267 | 0.309  | 0.333 |
| PPO (Unsloth+LoRA) | 0.334 | 0.253  | **0.352** |

A few things worth pulling out of these numbers:

- **Easy is mostly a sanity check.** Even a uniformly-random policy reaches `0.887` because traffic stays well under capacity. The environment doesn't really start *testing* a policy until medium.
- **Medium is the difficulty cliff.** Random and DQN both crash within ~15 ticks once SLAs are introduced. LRU's `0.253` — a *much* smaller number than its `0.977` on easy — is the strongest passing score, and the gap reflects how much harder queue navigation gets when latency starts mattering.
- **The LLM / PPO numbers are short-horizon.** Both agents crash within ~25–30 ticks on `medium` and `hard` on the hosted backend; their headline score is buoyed upward by the small denominator. The exception is PPO on `medium`: it sustains 138 ticks and completes 12 requests before crashing — the best LLM-class number on that tier and a sign that the LoRA fine-tune learned something the base Qwen did not.
- **Parse failures are not the bottleneck.** Across all six hosted runs, the LLM and PPO agents each had **0 parse failures** — every model output was a valid action ID. The crashes are policy failures, not formatting failures.
- **The classical baseline is harder to beat than people assume.** LRU's `0.977` on easy and `0.253` on medium are the bars a learned policy actually has to clear. None of the learned agents shipped here clear them on `medium` end-to-end. The case for dynamic eviction is, for now, still mostly theoretical — which is exactly why the environment exists.

---

## The dashboard

A live dashboard is available at:

**[https://neural-paged-attention-dashboard-7q.vercel.app/](https://neural-paged-attention-dashboard-7q.vercel.app/)**

It is a thin client over the Hugging Face Space backend (`https://suryanshchattree-neural-paged-attention-env.hf.space/api`) and exists for one purpose: to make the trade-offs above visible while a policy is running, not just after it finishes.

### What the dashboard surfaces

- A **configuration panel** for the global environment knobs — deterministic traffic seed, GPU/CPU total blocks, tokens per block, and the max-tick cap per difficulty. A single **START BATCH SIMULATION** button kicks off all three tasks for the agents you've selected.
- A **multi-agent comparison** chart that plots cumulative score for every agent on the same time axis, so the moment one policy starts pulling away from another is visible mid-episode rather than only in the final number.
- An **agent focus** panel with live Tick, Score, Total Reward, current Action, GPU / CPU utilization, memory-pressure trend, and a yield-preempt indicator — the same eight fields the LLM agent sees in its prompt, so the dashboard is also a debugger for the LLM policy.
- Separate **Free Requests** and **VIP Requests** queue charts, which is where the SLA-miss penalties actually originate.
- A **live action / reward / tick log**, plus per-session `COMPLETED` / `CRASHED` badges that mirror the `crashed` field returned by the API.

### What it lets you actually see

The numbers in the table above are end-of-episode summaries. The dashboard adds the dynamics that produced them:

1. **Crashes are sudden, not gradual.** The cumulative-score line for Random and DQN flattens within seconds of starting medium / hard. The score table shows `0.001`; the dashboard shows the cliff.
2. **GPU utilization tends to climb in concentrated 5–10 tick bursts**, not slow drift. Policies that don't pre-empt aggressively lose the window — which is why the LRU "swap oldest at 85%" rule does so much better than its simplicity suggests.
3. **Throughput and latency trade off in real time.** During traffic spikes you can watch LRU's cumulative score stall (it's paying the swap tax to keep VIPs admitted) while the VIP queue stays empty — the score-cost of preserving SLA visible side by side with the SLA itself.
4. **Each policy has a fingerprint.** Random's action stream is uniform, LRU concentrates on `Swap Oldest` / `Evict Oldest` once GPU exceeds 85%, the Q-learning agent over-issues `Reject` actions, and DQN's behavior shifts character around the same memory thresholds LRU uses — which is why their failure modes differ so much across tasks.
5. **Reward shape ≠ score shape.** Cumulative reward can be deeply negative while the final composite score is still `> 0.2`. That divergence is informative when comparing a "crashed but completed many requests" run against an "early-terminated" one.

The dashboard is not the answer to whether dynamic eviction beats static — it is the surface on which that question becomes legible.

---

## Where This Goes

The immediate value is benchmarking: a place to run policies side by side and understand *why* one outperforms another, not just *by how much*.

Longer-term, environments like this one could serve as training grounds for policies that are then validated in shadow mode against real servers. That gap — between simulator and production — is the important next step, and it requires real traffic data and hardware-in-the-loop testing that is beyond the scope of this project.

The environment is open-source, built on OpenEnv, and runnable in the HuggingFace Space linked below.

---

## Links

- 🌐 **Live dashboard:** https://neural-paged-attention-dashboard-7q.vercel.app/
- 🤗 **HuggingFace Space (backend):** https://suryanshchattree-neural-paged-attention-env.hf.space
- 📄 **README (technical details):** [`README.md`](./README.md)
- 📓 **Training Notebook:** https://huggingface.co/spaces/suryanshchattree/npa-ppo-train
