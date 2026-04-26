"""
train_ppo.py — Unsloth + PPO Reinforcement Learning for the LLM Agent.

IMPORTANT HARDWARE NOTE:
This script relies on `unsloth` which requires an NVIDIA GPU (Linux or Windows WSL).
You CANNOT run this script natively on an Apple Silicon Mac.
Please upload your project to Google Colab, RunPod, or a similar cloud GPU service to train.

Usage on Colab (T4 / A100):
  pip install -r requirements.txt
  pip install unsloth trl peft
  python3 agents/UnslothPPOAgent/train_ppo.py
"""

import os
import sys
import re
import torch
import random

# Path bootstrap to load environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from server.environment import KVCacheEnvironment
from agents.LLMAgent.prompts import build_user_prompt, SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024

def parse_action(text: str) -> int | None:
    matches = re.findall(r"\b(\d{1,2})\b", text)
    for m in matches:
        val = int(m)
        if 0 <= val <= 17:
            return val
    return None

def main():
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[!] Missing dependencies. Please run: pip install unsloth")
        sys.exit(1)

    print("="*60)
    print("  UNSLOTH REINFORCE (POLICY GRADIENT) TRAINING INITIALIZATION")
    print("="*60)

    # 1. Load Unsloth Model in 4-bit for extreme memory efficiency
    print("[*] Loading Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Apply LoRA Adapters
    print("[*] Injecting LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Setup Pure PyTorch Optimizer (Bypassing TRL completely)
    print("[*] Setting up AdamW Optimizer for Policy Gradient...")
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 4. Environment Rollout & Training Loop
    env = KVCacheEnvironment()
    
    TRAINING_EPISODES = 15  # Benchmark level
    GRADIENT_ACCUMULATION_STEPS = 4
    
    print("\n[*] Starting REINFORCE Rollouts...\n")
    running_baseline = 0.0
    
    for episode in range(TRAINING_EPISODES):
        current_task = random.choice(["easy", "medium", "hard"])
        obs_obj = env.reset(current_task)
        if obs_obj is None:
            continue
            
        obs = obs_obj.to_array()
        done = False
        tick = 0
        total_reward = 0.0
        optimizer.zero_grad()
        
        while not done and tick < 300: # Increased horizon to experience spikes
            # Format observation into prompt
            user_msg = build_user_prompt(obs, tick)
            
            # Use Qwen Chat Template
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
                
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            
            # --- PHASE 1: Generate Action (No Gradients) ---
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=8, 
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.7, 
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            
            # Clone to detach from no_grad/inference mode so Autograd doesn't complain during backprop
            output_ids = output_ids.clone()
            response_ids = output_ids[0][prompt_len:].clone()
            
            # Skip if model produced empty response
            if len(response_ids) == 0:
                print(f"    [!] Empty generation at tick {tick}. Skipping update.")
                action = 17; reward = -5.0
            else:
                response_str = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                # --- PHASE 2: Step Environment ---
                action = parse_action(response_str)
                if action is None:
                    action = 17 # Idle
                    reward = -5.0 # Harsh penalty for hallucinating / bad formatting
                    print(f"    [!] Hallucination penalty applied. Output: '{response_str}'")
                else:
                    obs_obj, reward, done, _ = env.step(action)
                    obs = obs_obj.to_array() if obs_obj else None
                
            total_reward += reward
            
            # --- PHASE 3: Policy Gradient Update (WITH Gradients) ---
            if len(response_ids) > 0:
                # Forward pass on the full sequence to get logits (disable cache for training)
                outputs = model(output_ids, use_cache=False)
                
                # Logits shifted by 1 to align with labels
                # We want the logits that predict the response_ids
                logits = outputs.logits[0, prompt_len-1 : -1, :] # Shape: (response_len, vocab_size)
                
                # Compute log probabilities of the actual generated tokens
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Gather the log prob of the specific chosen token IDs
                selected_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
                
                # Sum log probs over the response sequence
                seq_log_prob = selected_log_probs.sum()
                
                # Calculate Advantage using a running baseline to reduce variance
                advantage = reward - running_baseline
                running_baseline = 0.9 * running_baseline + 0.1 * reward
                
                # REINFORCE Objective: Loss = -log(prob) * Advantage
                loss = -seq_log_prob * advantage
                
                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (tick + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent gradient explosions
                    optimizer.step()
                    optimizer.zero_grad()
                
                loss_val = loss.item() * GRADIENT_ACCUMULATION_STEPS
            else:
                loss_val = 0.0
            
            if tick % 10 == 0:
                adv_val = advantage if len(response_ids) > 0 else 0.0
                print(f"[EP {episode} | {current_task.upper():<6s} | T{tick:3d}] Action: {action:2d} | Reward: {reward:5.2f} | Adv: {adv_val:6.2f} | Loss: {loss_val:.4f}")
            
            tick += 1

        print(f"\n[✓] Episode {episode} Complete | Total Reward: {total_reward:.2f}\n" + "-"*50)

    # 5. Save the trained LoRA adapter
    output_dir = os.path.join(os.path.dirname(__file__), "ppo_lora_agent")
    print(f"[*] Training complete! Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[*] Done. You can now use ppo_agent.py on your Mac to run inference with this adapter!")

if __name__ == "__main__":
    main()
