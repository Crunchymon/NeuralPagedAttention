"""
ppo_agent.py — Locally-running PPO fine-tuned agent for the Neural PagedAttention environment.

This agent will look for the `ppo_lora_agent` adapter saved by `train_ppo.py`.
If it finds the adapter, it loads it (using MLX if on Mac, or HuggingFace PEFT on CUDA).
"""

import os
import re
import sys
import time
import uuid
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models import ACTION_MAP
from server.environment import KVCacheEnvironment
from server.env_components.scoring import compute_final_score
from agents.LLMAgent.prompts import SYSTEM_PROMPT, build_user_prompt
from agents.LLMAgent.llm import LocalEnv

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "ppo_lora_agent")
MAX_NEW_TOKENS = 8
VALID_ACTIONS = set(range(18))

class PPOAgent:
    def __init__(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.parse_failures = 0
        self.use_mlx = False
        
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"[*] PPOAgent: CUDA detected")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.use_mlx = True
            print("[*] PPOAgent: Apple MPS detected — using MLX")
        else:
            self.device = "cpu"
            print("[*] PPOAgent: CPU mode")

        if self.use_mlx:
            import mlx_lm
            from mlx_lm.tuner.utils import load_adapters
            print(f"[*] Loading base model and tokenizer via mlx_lm...")
            
            # Check if adapter exists
            if os.path.exists(ADAPTER_PATH):
                print(f"[*] Found LoRA Adapter at {ADAPTER_PATH}! Loading and patching MLX model...")
                self.model, self.tokenizer = mlx_lm.load(MODEL_NAME)
                
                # Monkey-patch config to fix mlx-lm LoRA bug for Qwen2
                if hasattr(self.model, "args"):
                    if not hasattr(self.model.args, "num_layers") and hasattr(self.model.args, "num_hidden_layers"):
                        setattr(self.model.args, "num_layers", self.model.args.num_hidden_layers)
                
                # Inject adapters
                self.model = load_adapters(self.model, ADAPTER_PATH)
            else:
                print(f"[!] Warning: No adapter found at {ADAPTER_PATH}. Falling back to base model.")
                self.model, self.tokenizer = mlx_lm.load(MODEL_NAME)
                
            print("[*] PPOAgent: MLX Model ready.\n")
        else:
            from peft import PeftModel
            print(f"[*] Loading PyTorch base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if os.path.exists(ADAPTER_PATH):
                print(f"[*] Found LoRA Adapter! Injecting...")
                self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
            else:
                print(f"[!] Warning: No adapter found at {ADAPTER_PATH}.")
                
            self.model.eval()
            print("[*] PPOAgent: PyTorch Model ready.\n")

    def select_action(self, obs: list[float], tick: int) -> int:
        user_msg = build_user_prompt(obs, tick)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{SYSTEM_PROMPT}\n\n{user_msg}"

        t_start = time.time()
        try:
            if self.use_mlx:
                import mlx_lm
                response = mlx_lm.generate(
                    self.model,
                    self.tokenizer,
                    prompt=text,
                    max_tokens=MAX_NEW_TOKENS,
                    verbose=False
                ).strip()
            else:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=None,
                    )
                new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            print(f"[!] LLM generate error: {exc}")
            self.parse_failures += 1
            return 17

        elapsed = time.time() - t_start
        action = self._parse_action(response)
        
        if action is None:
            self.parse_failures += 1
            action = 17 # heuristic
            print(f"[!] Tick {tick}: parse failed ('{response}') → heuristic {action}  [{elapsed:.1f}s]")
        else:
            print(f"[*] Tick {tick}: PPO Agent → action {action} ({ACTION_MAP.get(action)})  [{elapsed:.1f}s]")

        return action

    def _parse_action(self, text: str) -> int | None:
        stripped = text.strip()
        if stripped.isdigit():
            val = int(stripped)
            if val in VALID_ACTIONS:
                return val
        matches = re.findall(r"\b(\d{1,2})\b", stripped)
        for m in matches:
            val = int(m)
            if val in VALID_ACTIONS:
                return val
        return None

def run_sim(task: str | None = None, ticks: int | None = None):
    env = LocalEnv()
    agent = PPOAgent()

    print("\n" + "=" * 60)
    print("  UNSLOTH PPO AGENT SIMULATION")
    print("=" * 60 + "\n")

    tasks_to_run = ["easy", "medium", "hard"] if task is None else [task]
    
    for current_task in tasks_to_run:
        obs = env.reset(current_task)
        if obs is None: continue
        
        total_reward = 0.0
        ticks_run = 0
        done = False

        while not done:
            # Pre-step batch admit
            gpu_util, free_q, vip_q = obs[0], obs[3], obs[4]
            if gpu_util < 0.85 and (vip_q > 0 or free_q > 0):
                pct = env.gpu_free_pct()
                if vip_q > 0: env.admit_batch("vip", pct)
                if free_q > 0: env.admit_batch("free", pct)

            action = agent.select_action(obs, tick=ticks_run)
            action_name = ACTION_MAP.get(action, "Unknown")

            obs, reward, done, _ = env.step(action)
            if obs is None: break

            total_reward += reward
            ticks_run += 1

            print(
                f"[{current_task.upper()} T{ticks_run:4}] "
                f"{action_name:35} | R {reward:+7.2f} | Total {total_reward:9.2f} | GPU {obs[0]:.2f}"
            )

        print(f"\n[✓] {current_task.upper()} Score: {total_reward:.2f} | Parse failures: {agent.parse_failures}\n" + "-" * 60)

if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_sim(task=task_arg)
