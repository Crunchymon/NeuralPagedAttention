---
title: NeuralPagedAttention — PPO 4xL4
emoji: 🚀
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
---

# Train PPO on Hugging Face Spaces — 4x Nvidia L4 (multi-GPU DDP)

This Space trains the LLM PPO agent for `NeuralPagedAttention` across **all four L4 GPUs**
using `accelerate launch` over [`agents/UnslothPPOAgent/train_ppo_ddp.py`](../agents/UnslothPPOAgent/train_ppo_ddp.py),
streams the training log into Gradio, and (optionally) uploads the resulting LoRA adapter
to a model repo on the Hub.

The default base model is **`Qwen/Qwen2.5-7B-Instruct`** (4-bit BNB + LoRA r=16, alpha=32),
the strongest model that fits comfortably on a 24 GB L4 under DDP without sharding.

> The single-GPU Unsloth path in [`agents/UnslothPPOAgent/train_ppo.py`](../agents/UnslothPPOAgent/train_ppo.py)
> is unchanged; this Space replaces Unsloth (single-GPU only) with HF Transformers + PEFT + Accelerate
> for a true 4xGPU run.

## 1. Create the Space

1. Push this whole repository to GitHub (or duplicate on Hugging Face).
2. [Create a new Space](https://huggingface.co/new-space) with **SDK = Docker**.
3. **Settings → Build options:**
   - **Dockerfile path:** `hf_ppo_space/Dockerfile`
   - **Build context:** repository **root** (the parent of `hf_ppo_space/`)
4. **Settings → Hardware:** select **4x Nvidia L4 (small)**.
   GPU Spaces are billed **per minute** — pause when done.

## 2. Configure secrets and variables

**Secrets** (Settings → Repository secrets):

| Name | Purpose |
| --- | --- |
| `HF_TOKEN` | Required to upload the adapter and to download gated models (Qwen). Use a token with **write** access if `OUTPUT_REPO_ID` is set. |

**Variables** (Settings → Variables):

| Name | Default | Meaning |
| --- | --- | --- |
| `AUTO_TRAIN` | `1` | Start training automatically on Space boot. Set `0` to start manually from the UI. |
| `PPO_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Base model. Drop to `Qwen/Qwen2.5-3B-Instruct` for fast iteration; push to `Qwen/Qwen2.5-14B-Instruct` as a tight upper bound. |
| `PPO_EPISODES` | `80` | Total RL episodes. |
| `PPO_LR` | `2e-5` | AdamW learning rate (LoRA only). |
| `PPO_ENTROPY` | `0.04` | Initial entropy coefficient (decayed linearly). |
| `PPO_MAX_TICKS` | *(empty)* | Optional per-episode tick cap (e.g. `1200`). Empty = use task default. |
| `PPO_CURRICULUM` | *(empty)* | Episodes pinned to `easy` at the start. Empty = 25% of total. |
| `PPO_SAVE_EVERY` | `0` | Save adapter every N episodes (0 = end only). |
| `PPO_NO_4BIT` | `0` | Set `1` to disable BNB-4bit and use bf16 LoRA only (memory fallback). |
| `PPO_SEED` | `42` | RNG seed. |
| `OUTPUT_REPO_ID` | *(empty)* | Hub model repo to push the trained adapter **and training logs**, e.g. `your-user/npa-ppo`. Create the empty repo first. |
| `LOG_UPLOAD_INTERVAL_SEC` | `60` | Seconds between in-flight log snapshot uploads to the Hub repo. |

## 3. What you get

After a successful run, [`agents/UnslothPPOAgent/ppo_lora_agent/`](../agents/UnslothPPOAgent/ppo_lora_agent)
contains:

- `adapter_model.safetensors` (LoRA weights)
- `adapter_config.json` (records `base_model_name_or_path` so inference can pick the right base)
- tokenizer files

If `OUTPUT_REPO_ID` is set, the same folder also lands on the Hub at `OUTPUT_REPO_ID/ppo_lora_agent/`,
**along with the full training log under** `OUTPUT_REPO_ID/logs/<RUN_ID>/`:

| File | Contents |
| --- | --- |
| `train.log` | Combined stdout from all 4 ranks, including per-tick lines, per-episode summaries, schedules, and any errors. Refreshed every 60 s during training, plus one guaranteed final upload at run end. |
| `metrics.jsonl` | One JSON object per episode (`episode`, `task`, `ticks`, `crashed`, `final_score`, `completed/arrived`, `cum_step_return`, `entropy_coef`, `temperature`) plus `run_start` / `run_end` markers. Easy to load with `pandas.read_json(..., lines=True)`. |

`<RUN_ID>` is the UTC timestamp at which the run started (e.g. `2026-04-26T12-30-00Z`),
so re-running the Space appends a new `logs/<RUN_ID>/` folder rather than overwriting
the previous run's log.

Pull the adapter down and place it next to [`ppo_agent.py`](../agents/UnslothPPOAgent/ppo_agent.py)
to evaluate locally:

```bash
python agents/UnslothPPOAgent/ppo_agent.py easy
```

(`ppo_agent.py` reads the base model name from the adapter config — no code change needed when
you train against a different base size.)

## 4. Why this layout

- **DDP, not FSDP.** Qwen2.5-7B in 4-bit is ~5 GB; on each 24 GB L4 we fit weights, bf16
  activations, LoRA grads/Adam state, and the tiny KV cache (only ~8 generated tokens per tick)
  with comfortable headroom. Sharding only adds latency.
- **Per-rank seeded environments.** Every rank runs an independent `KVCacheEnvironment`
  with `TRAFFIC_SEED = base + rank * c`, so the four GPUs co-train on uncorrelated rollouts —
  a real 4× effective batch.
- **Curriculum + entropy/temperature decay + reward clip.** Mitigates the early-termination
  pattern (24-tick crashes on hard tasks) seen with the single-GPU run.
- **Rank-0-only save and upload.** Avoids race conditions on the adapter folder.

## 5. Local Docker test

From repository root:

```bash
docker build -f hf_ppo_space/Dockerfile -t npa-ppo-ddp .
docker run --gpus all -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e OUTPUT_REPO_ID=your-user/npa-ppo \
  -e PPO_EPISODES=2 -e PPO_MAX_TICKS=200 \
  npa-ppo-ddp
```

Open `http://localhost:7860`. With fewer than 4 GPUs, `accelerate` will still launch — the
config sets `num_processes: 4`, so for local single-GPU smoke tests pass
`-e ACCEL_CONFIG=` and rely on `accelerate launch` defaults, or run the trainer directly:

```bash
python agents/UnslothPPOAgent/train_ppo_ddp.py --episodes 1 --max-ticks 100
```

## 6. Billing

GPU Spaces are billed **per minute** while running. **Pause** the Space the moment training
finishes (or set `PPO_EPISODES` conservatively). Uploads happen after the trainer exits;
the log will show `Upload finished.` before the Space becomes idle.
