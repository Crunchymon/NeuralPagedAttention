import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "agents/UnslothPPOAgent/ppo_lora_agent"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map=None)

print("Injecting adapter...")
import os
mlx_file = os.path.join(ADAPTER_PATH, "adapters.safetensors")
peft_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
if os.path.exists(mlx_file) and not os.path.exists(peft_file):
    print(f"Creating symlink {peft_file} -> adapters.safetensors")
    os.symlink("adapters.safetensors", peft_file)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("Done!")
