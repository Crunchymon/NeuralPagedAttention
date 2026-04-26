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
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
    except ImportError:
        print("[!] Missing dependencies. Please run: pip install unsloth trl peft")
        sys.exit(1)

    print("="*60)
    print("  UNSLOTH PPO TRAINING INITIALIZATION")
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

    # 3. Setup TRL PPO framework
    print("[*] Setting up PPO Value Head & Reference Model...")
    # PPO needs a critic (Value Head) to evaluate how good states are
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    ref_model = create_reference_model(ppo_model)
    
    config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # 4. Environment Rollout & Training Loop
    env = KVCacheEnvironment()
    
    TRAINING_EPISODES = 5  # Start small for testing
    
    print("\n[*] Starting PPO Rollouts...\n")
    for episode in range(TRAINING_EPISODES):
        obs_obj = env.reset("easy")
        if obs_obj is None:
            continue
            
        obs = obs_obj.to_array()
        done = False
        tick = 0
        total_reward = 0.0
        
        while not done and tick < 100: # Limit ticks per episode for faster RL cycles
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
                
            inputs = tokenizer(prompt, return_tensors="pt").to(ppo_trainer.accelerator.device)
            query_tensor = inputs["input_ids"][0]
            
            # Generate action via policy
            generation_kwargs = {"max_new_tokens": 8, "do_sample": True, "top_p": 0.9, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id}
            response_tensor = ppo_trainer.generate(query_tensor.unsqueeze(0), **generation_kwargs)
            response_tensor = response_tensor.squeeze()[len(query_tensor):]
            response_str = tokenizer.decode(response_tensor, skip_special_tokens=True).strip()
            
            # Step Environment
            action = parse_action(response_str)
            if action is None:
                action = 17 # Idle
                reward = -5.0 # Harsh penalty for hallucinating / bad formatting
                print(f"    [!] Hallucination penalty applied. Output: '{response_str}'")
            else:
                obs_obj, reward, done, _ = env.step(action)
                obs = obs_obj.to_array() if obs_obj else None
                
            total_reward += reward
            
            # Run PPO Update
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(ppo_trainer.accelerator.device)
            stats = ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
            
            if tick % 10 == 0:
                print(f"[EP {episode} | T{tick:3d}] Action: {action:2d} | Reward: {reward:5.2f} | PPO Loss: {stats['ppo/loss/total']:.4f}")
            
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
