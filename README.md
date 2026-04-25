---
title: Neural Paged Attention
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: Docker
sdk_version: "4.36.1"
app_file: app.py
pinned: false
---
# Neural PagedAttention

RL environment for training AI agents to manage GPU KV Cache in LLM inference servers.

## Motivation
Modern LLM servers are bottlenecked by memory. Standard heuristics like LRU
fail under bimodal traffic because they cannot predict future requests or
distinguish VIP from Free users. This environment trains RL agents to do both.

## Environment Description
A deterministic Python simulator of PagedAttention hardware.
- 1 Tick = 1 token generated across all active requests
- GPU: 10,000 blocks (160,000 tokens)
- CPU: 50,000 blocks (800,000 tokens) — swap target
- Traffic: 80% Chatters, 20% Power Users, 30% Returning Users

## Observation Space
20-dimensional normalized float array [0.0, 1.0] (trend: [-1.0, 1.0]).

| # | Field | Description |
|---|-------|-------------|
| 1 | `gpu_utilization_pct` | Fraction of GPU blocks in use |
| 2 | `cpu_utilization_pct` | Fraction of CPU blocks in use |
| 3 | `memory_pressure_trend` | GPU utilization slope over last 5 ticks (negative = falling) |
| 4 | `free_queue_pressure` | Free queue length / max (100) |
| 5 | `vip_queue_pressure` | VIP queue length / max (50) |
| 6 | `free_max_wait_time_pct` | Oldest Free request wait / SLA threshold |
| 7 | `vip_max_wait_time_pct` | Oldest VIP request wait / SLA threshold |
| 8 | `yield_preempt_active` | Largest active request size / GPU capacity |
| 9 | `free_size_max` | Largest idle Free GPU cache / GPU capacity |
| 10 | `free_size_mean` | Mean idle Free GPU cache size |
| 11 | `free_size_std_dev` | Std dev of idle Free GPU cache sizes |
| 12 | `vip_size_max` | Largest idle VIP GPU cache / GPU capacity |
| 13 | `vip_size_mean` | Mean idle VIP GPU cache size |
| 14 | `vip_size_std_dev` | Std dev of idle VIP GPU cache sizes |
| 15 | `free_age_max` | Oldest idle Free GPU cache age (normalized to 500 ticks) |
| 16 | `free_age_mean` | Mean idle Free GPU cache age |
| 17 | `free_age_std_dev` | Std dev of idle Free GPU cache ages |
| 18 | `vip_age_max` | Oldest idle VIP GPU cache age |
| 19 | `vip_age_mean` | Mean idle VIP GPU cache age |
| 20 | `vip_age_std_dev` | Std dev of idle VIP GPU cache ages |

## Action Space
Discrete(18), indices 0–17.

| ID | Action |
|----|--------|
| 0 | Evict Largest Free cache (idle GPU only) |
| 1 | Evict Largest VIP cache (idle GPU only) |
| 2 | Evict Oldest Free cache (idle GPU only) |
| 3 | Evict Oldest VIP cache (idle GPU only) |
| 4 | Swap Largest Free cache GPU→CPU |
| 5 | Swap Largest VIP cache GPU→CPU |
| 6 | Swap Oldest Free cache GPU→CPU |
| 7 | Swap Oldest VIP cache GPU→CPU |
| 8 | Admit next Free user from queue |
| 9 | Admit next VIP user from queue |
| 10 | Reject next Free user (penalty if GPU has space) |
| 11 | Reject next VIP user (penalty if GPU has space) |
| 12 | Preempt & Shred Largest Active Free request |
| 13 | Preempt & Shred Largest Active VIP request |
| 14 | Preempt & Swap Largest Active Free → CPU |
| 15 | Preempt & Swap Largest Active VIP → CPU |
| 16 | Garbage Collect (delete idle Free CPU caches > 200 ticks) |
| 17 | Do Nothing |

## Tasks
| Task   | Traffic     | SLA Free | SLA VIP | Max Ticks |
|--------|-------------|----------|---------|-----------|
| easy   | 1 req/tick  | None     | None    | 2,000     |
| medium | Day/night   | 100      | 50      | 5,000     |
| hard   | Viral spikes| 50       | 25      | 10,000    |

## Baseline Scores
| Task   | Random Agent | LLM Baseline |
|--------|-------------|--------------|
| easy   | TBD         | TBD          |
| medium | TBD         | TBD          |
| hard   | TBD         | TBD          |

## Setup
```bash
docker build -t neural-paged-attention ./server
docker run -p 7860:7860 neural-paged-attention
```

## Running Inference
```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```
