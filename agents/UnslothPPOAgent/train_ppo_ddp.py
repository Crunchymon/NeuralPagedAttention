"""
train_ppo_ddp.py — Multi-GPU REINFORCE-style PPO trainer for the LLM agent.

Designed for a Hugging Face Space configured with 4x Nvidia L4 (or any DDP
multi-GPU box). Replaces the single-GPU Unsloth path in train_ppo.py with:

  * HF Transformers + bitsandbytes 4-bit (NF4 + double-quant)
  * PEFT LoRA (r=16, alpha=32) on q/k/v/o/gate/up/down projections
  * Accelerate DDP — each rank owns its own GPU, model copy, and KVCacheEnvironment
  * bf16 mixed precision compute
  * Per-rank seeded traffic so the 4 GPUs co-train on uncorrelated trajectories
  * Curriculum (first ~25% of episodes pinned to 'easy'), entropy + temperature decay,
    chunked discounted returns, gradient clipping, reward clipping
  * Rank-0-only adapter save (compatible with agents/UnslothPPOAgent/ppo_agent.py)

Default base model is Qwen/Qwen2.5-7B-Instruct (the strongest model that fits
comfortably on a 24 GB L4 under DDP + 4-bit + LoRA + bf16). Override with
--model-name (Qwen/Qwen2.5-3B-Instruct for fast iteration; Qwen/Qwen2.5-14B-Instruct
as a tight upper bound).

Launch via accelerate:

    accelerate launch --config_file hf_ppo_space/accelerate_config.yaml \\
        agents/UnslothPPOAgent/train_ppo_ddp.py \\
        --episodes 80 --lr 2e-5 --entropy-coef 0.04

For local single-GPU debugging:

    python agents/UnslothPPOAgent/train_ppo_ddp.py --episodes 2 --max-ticks 200
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

import torch
import torch.nn.functional as F

try:
    import transformers

    transformers.logging.set_verbosity_error()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import server.env_components.constants as constants  # noqa: E402
from agents.LLMAgent.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from server.env_components.scoring import compute_final_score  # noqa: E402
from server.environment import KVCacheEnvironment  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 1024
PARSE_FAIL_EXTRA = -2.0
CHUNK_SIZE = 24
GAMMA = 0.99
MAX_GRAD_NORM = 1.0
GEN_EXTRA_TOKENS = 8
TASKS = ["easy", "medium", "hard"]


def parse_action(text: str) -> int | None:
    matches = re.findall(r"\b(\d{1,2})\b", text)
    for m in matches:
        v = int(m)
        if 0 <= v <= 17:
            return v
    return None


def pre_step_admit(env: KVCacheEnvironment) -> None:
    """Same heuristic as train_ppo.py / ppo_agent.py: top up the GPU when slack exists."""
    if env.ledger.gpu_utilization() < 0.85 and (env.vip_queue or env.free_queue):
        pct = max(0.0, (constants.GPU_TOTAL_BLOCKS - env.ledger.gpu_used) / constants.GPU_TOTAL_BLOCKS)
        if env.vip_queue:
            env.admit_batch("vip", pct)
        if env.free_queue:
            env.admit_batch("free", pct)


def compute_logprob_entropy(model, output_ids: torch.Tensor, prompt_len: int):
    """Forward through the (DDP-wrapped) model, returning (seq_log_prob, mean_entropy)."""
    out = model(output_ids, use_cache=False)
    logits = out.logits[0, prompt_len - 1 : -1, :]
    response_ids = output_ids[0, prompt_len:].clone()
    if response_ids.numel() == 0:
        return None, None
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    selected = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    return selected.sum(), entropy.mean()


def flush_chunk(buf, model, optimizer, accelerator, entropy_coef: float) -> float:
    """Discounted normalized returns → policy-gradient loss → DDP backward."""
    if not buf:
        return 0.0
    rewards = [b[2] for b in buf]
    R = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        R = float(r) + GAMMA * R
        returns.insert(0, R)
    device = buf[0][0].device
    returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std().clamp(min=1e-6))

    loss = torch.zeros((), device=device)
    ents: list[torch.Tensor] = []
    for i, (lp, ent, _) in enumerate(buf):
        loss = loss - returns_t[i].detach() * lp
        ents.append(ent)
    loss = loss / len(buf) - entropy_coef * torch.stack(ents).mean()

    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help=f"HF model id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.04)
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Cap ticks per episode (0 = use task default).")
    parser.add_argument("--curriculum-episodes", type=int, default=-1,
                        help="Episodes pinned to 'easy' at the start (-1 = 25%% of total).")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save adapter every N episodes (0 = only at end).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable BNB 4-bit; use bf16 LoRA only (memory fallback).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Exit after imports + accelerator init (smoke test).")
    args = parser.parse_args()

    from accelerate import Accelerator, DistributedDataParallelKwargs
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world = accelerator.num_processes

    rank_seed = args.seed + rank * 9973
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)

    if is_main:
        print("=" * 64)
        print(f"  PPO DDP TRAINER — model={args.model_name} | world={world}")
        print("=" * 64)

    if args.dry_run:
        if is_main:
            print(f"[*] dry-run OK | rank={rank}/{world} | device={accelerator.device}")
        return

    if is_main:
        print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model_kwargs: dict = dict(
        trust_remote_code=True,
        device_map={"": accelerator.local_process_index},
        attn_implementation="sdpa",
    )
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if is_main:
        print(f"[*] Loading base model on rank {rank} → cuda:{accelerator.local_process_index} "
              f"(4bit={bnb_config is not None})")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    if is_main:
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    env = KVCacheEnvironment()

    if args.curriculum_episodes < 0:
        curriculum_eps = max(1, args.episodes // 4)
    else:
        curriculum_eps = args.curriculum_episodes

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "ppo_lora_agent")

    if is_main:
        print(f"[*] Episodes={args.episodes} | curriculum_easy={curriculum_eps} | "
              f"lr={args.lr} | chunk={CHUNK_SIZE} | seed_rank0={args.seed}")
        print(f"[*] Output dir: {output_dir}\n")

    for episode in range(args.episodes):
        # Same task across all ranks (deterministic from seed+episode), uncorrelated traffic per rank.
        task_rng = random.Random(args.seed * 100003 + episode)
        current_task = "easy" if episode < curriculum_eps else task_rng.choice(TASKS)

        constants.TRAFFIC_SEED = (args.seed * 7 + episode * 31 + rank) & 0x7FFFFFFF
        obs_obj = env.reset(current_task)
        if obs_obj is None:
            continue

        max_t = env.config["max_ticks"]
        if args.max_ticks > 0:
            max_t = min(max_t, args.max_ticks)

        progress = episode / max(args.episodes, 1)
        entropy_coef = max(0.005, args.entropy_coef * (1.0 - progress))
        temperature = max(0.6, 0.85 - 0.25 * progress)

        obs = obs_obj.to_array()
        done = False
        tick = 0
        total_reward = 0.0
        chunk_buf: list[tuple[torch.Tensor, torch.Tensor, float]] = []

        optimizer.zero_grad(set_to_none=True)

        # Generation bypasses the DDP wrapper to avoid spurious all-reduces;
        # gradient-tracking forwards still go through the wrapped `model(...)`.
        gen_model = accelerator.unwrap_model(model)

        while not done and tick < max_t:
            pre_step_admit(env)
            user_msg = build_user_prompt(obs, tick)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(accelerator.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = gen_model.generate(
                    **inputs,
                    max_length=prompt_len + GEN_EXTRA_TOKENS,
                    do_sample=True,
                    top_p=0.92,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            output_ids = output_ids.clone()
            response_ids = output_ids[0, prompt_len:].clone()

            parse_penalty = 0.0
            if response_ids.numel() == 0:
                action = 17
                parse_penalty = PARSE_FAIL_EXTRA
            else:
                resp_str = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                parsed = parse_action(resp_str)
                if parsed is None:
                    action = 17
                    parse_penalty = PARSE_FAIL_EXTRA
                else:
                    action = parsed

            seq_lp, ent = compute_logprob_entropy(model, output_ids, prompt_len)
            if seq_lp is None:
                obs_obj, reward, done, _ = env.step(action)
                reward = float(reward) + parse_penalty
                total_reward += reward
                obs = obs_obj.to_array() if obs_obj else obs
                tick += 1
                continue

            obs_obj, reward, done, _ = env.step(action)
            reward = float(reward) + parse_penalty
            obs = obs_obj.to_array() if obs_obj else obs
            total_reward += reward

            r_clip = max(-40.0, min(40.0, reward))
            chunk_buf.append((seq_lp, ent, r_clip))

            if len(chunk_buf) >= CHUNK_SIZE:
                flush_chunk(chunk_buf, model, optimizer, accelerator, entropy_coef)
                chunk_buf = []

            if is_main and tick % 24 == 0:
                mean_r = total_reward / max(1, tick)
                print(
                    f"[EP {episode:3} | {current_task.upper():<6} | T{tick:5}] "
                    f"a={action:2} | r={reward:6.2f} | sum_r={total_reward:8.1f} | "
                    f"mean_r={mean_r:6.3f} | ent={entropy_coef:.3f} | T={temperature:.2f}",
                    flush=True,
                )

            tick += 1

        if chunk_buf:
            flush_chunk(chunk_buf, model, optimizer, accelerator, entropy_coef)

        if is_main:
            final_score = compute_final_score(
                task=current_task,
                total_completed=env.total_completed,
                total_arrived=env.total_arrived,
                per_request_fluency=env._per_request_fluency,
                total_cache_hits=env.total_cache_hits,
                total_returning_arrived=env.total_returning_arrived,
                total_swaps=env.total_swaps,
                total_actions=env.total_actions,
            )
            done_rate = env.total_completed / max(1, env.total_arrived)
            print(
                f"\n[EP {episode} done] task={current_task} | ticks={tick} | crashed={env.crashed}\n"
                f"    final_score={final_score:.4f} (0-1, higher=better) | "
                f"completed {env.total_completed}/{env.total_arrived} ({done_rate:.1%}) | "
                f"cum_step_return={total_reward:.1f}\n"
                + "-" * 64,
                flush=True,
            )

        if args.save_every > 0 and (episode + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if is_main:
                accelerator.unwrap_model(model).save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"[*] Checkpoint saved at episode {episode + 1} → {output_dir}", flush=True)

    accelerator.wait_for_everyone()
    if is_main:
        print(f"\n[*] Final save → {output_dir}")
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[*] Done. Run agents/UnslothPPOAgent/ppo_agent.py to evaluate.")


if __name__ == "__main__":
    main()
