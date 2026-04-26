"""
train_ppo.py — Policy gradient (REINFORCE-style) training for the LLM agent with Unsloth.

Requires NVIDIA GPU + unsloth for 4-bit training. On Apple Silicon, use a cloud GPU
or run `python train_ppo.py --dry-run` to validate imports only.

Improvements over the original:
  - Always calls env.step so the simulator never stalls on bad LLM output
  - Chunked discounted returns + one backward per chunk (stable, bounded memory)
  - Same batch-admit heuristic as LRU/PPO inference so the policy focuses on pressure actions
  - Configurable horizons, learning rate, entropy
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import server.env_components.constants as constants
from server.environment import KVCacheEnvironment
from agents.LLMAgent.prompts import SYSTEM_PROMPT, build_user_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
PARSE_FAIL_EXTRA = -2.0
CHUNK_SIZE = 24
GAMMA = 0.99
MAX_GRAD_NORM = 1.0


def parse_action(text: str) -> int | None:
    matches = re.findall(r"\b(\d{1,2})\b", text)
    for m in matches:
        val = int(m)
        if 0 <= val <= 17:
            return val
    return None


def pre_step_admit(env: KVCacheEnvironment) -> None:
    gpu_util = env.ledger.gpu_utilization()
    free_q = len(env.free_queue)
    vip_q = len(env.vip_queue)
    if gpu_util < 0.85 and (vip_q > 0 or free_q > 0):
        pct = max(0.0, (constants.GPU_TOTAL_BLOCKS - env.ledger.gpu_used) / constants.GPU_TOTAL_BLOCKS)
        if vip_q > 0:
            env.admit_batch("vip", pct)
        if free_q > 0:
            env.admit_batch("free", pct)


def compute_logprob_entropy(model, tokenizer, output_ids, prompt_len):
    """Forward pass on full sequence; return (seq_log_prob, mean_entropy)."""
    outputs = model(output_ids, use_cache=False)
    logits = outputs.logits[0, prompt_len - 1 : -1, :]
    response_ids = output_ids[0, prompt_len:].clone()
    if response_ids.numel() == 0:
        return None, None
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    selected = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    seq_log_prob = selected.sum()
    return seq_log_prob, entropy.mean()


def flush_chunk(
    buf: list[tuple[torch.Tensor, torch.Tensor, float]],
    model,
    optimizer,
    entropy_coef: float,
) -> float:
    """Discounted returns over chunk; policy loss = -sum R_t * log pi_t."""
    if not buf:
        return 0.0
    rewards = [b[2] for b in buf]
    R = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        R = float(r) + GAMMA * R
        returns.insert(0, R)
    returns_t = torch.tensor(returns, device=buf[0][0].device, dtype=torch.float32)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std().clamp(min=1e-6))

    loss = torch.zeros((), device=buf[0][0].device)
    ent_acc = []
    for i, (lp, ent, _) in enumerate(buf):
        loss = loss - returns_t[i].detach() * lp
        ent_acc.append(ent)
    loss = loss / len(buf) - entropy_coef * torch.stack(ent_acc).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-ticks", type=int, default=0, help="Cap ticks per episode (0 = use task default)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.04)
    parser.add_argument("--dry-run", action="store_true", help="Exit after imports (no GPU)")
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[!] Missing unsloth. Install on Linux/WSL/CUDA: pip install unsloth")
        sys.exit(1)

    if args.dry_run:
        print("[*] Dry run OK (unsloth import succeeded).")
        return

    print("=" * 60)
    print("  LLM POLICY GRADIENT (chunked REINFORCE + LoRA)")
    print("=" * 60)

    print("[*] Loading Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("[*] Injecting LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()

    env = KVCacheEnvironment()
    TRAINING_EPISODES = args.episodes

    print(f"\n[*] Starting training: {TRAINING_EPISODES} episodes | lr={args.lr} | chunk={CHUNK_SIZE}\n")

    for episode in range(TRAINING_EPISODES):
        current_task = random.choice(["easy", "medium", "hard"])
        obs_obj = env.reset(current_task)
        if obs_obj is None:
            continue

        max_t = env.config["max_ticks"]
        if args.max_ticks > 0:
            max_t = min(max_t, args.max_ticks)

        obs = obs_obj.to_array()
        done = False
        tick = 0
        total_reward = 0.0
        chunk_buf: list[tuple[torch.Tensor, torch.Tensor, float]] = []

        entropy_coef = max(0.01, float(args.entropy_coef) * (1.0 - episode / max(TRAINING_EPISODES, 1)))

        optimizer.zero_grad(set_to_none=True)

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

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.75,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )

            output_ids = output_ids.clone()
            response_ids = output_ids[0][prompt_len:].clone()

            parse_penalty = 0.0
            if response_ids.numel() == 0:
                action = 17
                parse_penalty = PARSE_FAIL_EXTRA
            else:
                response_str = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                parsed = parse_action(response_str)
                if parsed is None:
                    action = 17
                    parse_penalty = PARSE_FAIL_EXTRA
                else:
                    action = parsed

            seq_log_prob, ent_mean = compute_logprob_entropy(model, tokenizer, output_ids, prompt_len)
            if seq_log_prob is None:
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
            chunk_buf.append((seq_log_prob, ent_mean, r_clip))

            if len(chunk_buf) >= CHUNK_SIZE:
                flush_chunk(chunk_buf, model, optimizer, entropy_coef)
                chunk_buf = []

            if tick % 12 == 0:
                print(
                    f"[EP {episode} | {current_task.upper():<6} | T{tick:4}] "
                    f"action={action:2} | r={reward:6.2f} | total={total_reward:8.1f}"
                )

            tick += 1

        if chunk_buf:
            flush_chunk(chunk_buf, model, optimizer, entropy_coef)

        print(f"\n[✓] Episode {episode} | {current_task} | ticks={tick} | return={total_reward:.2f}\n" + "-" * 50)

    output_dir = os.path.join(os.path.dirname(__file__), "ppo_lora_agent")
    print(f"[*] Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[*] Done. Run ppo_agent.py to evaluate.")


if __name__ == "__main__":
    main()
