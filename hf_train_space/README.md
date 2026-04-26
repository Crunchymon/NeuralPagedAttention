---
title: NeuralPagedAttention — GPU training
emoji: 🏋️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
---

# Train on Hugging Face Spaces (GPU)

This Space runs **DQN** training inside your project checkout, streams logs in Gradio, and can **push checkpoints** to a Hub model repo.

## Create the Space

1. Push this **full repository** to GitHub (or duplicate on HF).
2. [Create a new Space](https://huggingface.co/new-space) → **Docker** SDK.
3. In Space **Settings → Build options**:
   - **Dockerfile path:** `hf_train_space/Dockerfile`
   - **Build context:** repository **root** (parent of `hf_train_space/`).
4. **Hardware:** enable a **GPU** (T4 / L4 / …) — CPU will be very slow for DQN and unusable for PPO.

For **PPO + Unsloth**, use `hf_train_space/Dockerfile.unsloth` instead (longer build; may need a larger GPU).

## Secrets and variables

**Repository secrets** (Settings → Secrets):

| Name | Purpose |
|------|--------|
| `HF_TOKEN` | Upload + gated models (e.g. Qwen). Use a token with **write** access if uploading. |

**Variables** (Settings → Variables):

| Name | Default | Meaning |
|------|---------|--------|
| `AUTO_TRAIN` | `1` | Start training when the Space boots. |
| `OUTPUT_REPO_ID` | *(empty)* | e.g. `your-user/npa-weights` — create an empty **model** repo first. |
| `DQN_EPISODES` | `500` | DQN marathon length. |
| `DQN_EVAL_EVERY` | `50` | Eval frequency. |
| `SEED` | `42` | RNG seed. |
| `DQN_FAST` | `0` | Set `1` for smoke test (short replay warmup). |
| `TRAIN_PPO` | `0` | Set `1` only with **Dockerfile.unsloth** image. |
| `PPO_EPISODES` | `80` | PPO episodes. |
| `PPO_MAX_TICKS` | *(empty)* | e.g. `1200` to cap ticks per episode. |

## After training

- If `OUTPUT_REPO_ID` is set, files appear on the Hub: `dqn_weights_best.pth`, `dqn_weights.pth`, and optionally `ppo_lora_agent/`.
- Otherwise copy artifacts from the Space **Files** tab or clone the Space repo after a commit (not automatic).

## Billing

GPU Spaces are billed **per minute** while running. **Pause** or downgrade hardware when idle.

## Local Docker test

From repository root:

```bash
docker build -f hf_train_space/Dockerfile -t npa-train .
docker run --gpus all -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e OUTPUT_REPO_ID=your-user/npa-weights \
  -e DQN_EPISODES=5 \
  -e DQN_FAST=1 \
  npa-train
```

Open `http://localhost:7860`.
