"""
Neural PagedAttention — OpenAI-Compatible Baseline Inference Script

Runs a language model against the Neural PagedAttention environment.
Follows the OpenEnv standard logging format strictly.

MANDATORY VARIABLES:
  OPENAI_API_KEY (or HF_TOKEN, GROQ_API_KEY, GEMINI_API_KEY)

OPTIONAL VARIABLES:
  API_BASE_URL (defaults to OpenAI, overridden appropriately for other providers)
  MODEL_NAME   (defaults to gpt-4o-mini)
  ENV_URL      (defaults to http://localhost:7860)
"""
import os
import sys
import textwrap
import time
from typing import Optional

import requests
from openai import OpenAI

# ── Credentials & Configuration ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if OPENAI_API_KEY:
    API_KEY      = OPENAI_API_KEY
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
elif HF_TOKEN:
    API_KEY      = HF_TOKEN
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
elif GROQ_API_KEY:
    API_KEY      = GROQ_API_KEY
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
elif GEMINI_API_KEY:
    API_KEY      = GEMINI_API_KEY
    API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL_NAME   = os.getenv("MODEL_NAME", "gemini-2.0-flash")
else:
    print("[ERROR] No API key found. Set OPENAI_API_KEY, HF_TOKEN, GROQ_API_KEY, or GEMINI_API_KEY.", file=sys.stderr)
    sys.exit(1)

ENV_URL   = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "neural-paged-attention"

# We will run all 3 tasks as requested by the grading checklist
TASKS = ["easy", "medium", "hard"]

MAX_STEPS   = 50
TEMPERATURE = 0.0  # Must be 0 for deterministic baseline
MAX_TOKENS  = 10   # Only need a single integer

ACTION_DESCRIPTIONS = textwrap.dedent("""
  0  = Evict Largest Free cache (idle GPU only)
  1  = Evict Largest VIP cache  (idle GPU only)
  2  = Evict Oldest Free cache  (idle GPU only)
  3  = Evict Oldest VIP cache   (idle GPU only)
  4  = Swap Largest Free cache  GPU->CPU
  5  = Swap Largest VIP cache   GPU->CPU
  6  = Swap Oldest Free cache   GPU->CPU
  7  = Swap Oldest VIP cache    GPU->CPU
  8  = Admit next Free user     from queue
  9  = Admit next VIP user      from queue
 10  = Reject next Free user    (heavy penalty if GPU has space)
 11  = Reject next VIP user     (heavy penalty if GPU has space)
 12  = Preempt & Shred          Largest Active Free request
 13  = Preempt & Shred          Largest Active VIP request
 14  = Preempt & Swap           Largest Active Free -> CPU
 15  = Preempt & Swap           Largest Active VIP -> CPU
 16  = Garbage Collect          (delete idle Free CPU caches > 200 ticks)
 17  = Do Nothing
""").strip()

SYSTEM_PROMPT = textwrap.dedent(f"""
You are an AI memory manager for a GPU cluster running LLM inference.
Your goal is to maximize throughput and keep VIP users happy (they are 3x more valuable).
Respond with ONLY a single integer 0-17. Do not include any explanations.

Available actions:
{ACTION_DESCRIPTIONS}
""").strip()


# ── Strict OpenEnv Logging Format ──────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP Environment Client ────────────────────────────────────────────────
def env_reset(task: str) -> dict:
    resp = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_step(action_id: int) -> dict:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    resp = requests.get(f"{ENV_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── LLM Action Logic ───────────────────────────────────────────────────────
_LAST_API_CALL = 0.0

def get_action(client: OpenAI, obs: dict, step: int) -> int:
    global _LAST_API_CALL
    # Rate limit pacing (useful for free tier keys)
    elapsed = time.time() - _LAST_API_CALL
    if elapsed < 2.1:
        time.sleep(2.1 - elapsed)
    _LAST_API_CALL = time.time()

    user_prompt = textwrap.dedent(f"""
    Step {step} — Current state:
    - GPU util:        {obs['gpu_utilization_pct']:.2f}
    - CPU util:        {obs['cpu_utilization_pct']:.2f}
    - Free queue reqs: {obs['total_free_req']:.0f}
    - VIP queue reqs:  {obs['total_vip_req']:.0f}
    - Total queue reqs:{obs['total_req']:.0f}
    - Free SLA risk:   {obs['free_max_wait_time_pct']:.2f}
    - VIP SLA risk:    {obs['vip_max_wait_time_pct']:.2f}

    Choose action (0-17):
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Parse the first integer from the response
        action_str = "".join(c for c in text.split()[0] if c.isdigit())
        if action_str:
            action_id = int(action_str)
            if 0 <= action_id <= 17:
                return action_id
        return 17  # Default to Do Nothing

    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", file=sys.stderr)
        return 17


def run_episode(client: OpenAI, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # 1. Reset Environment
        result = env_reset(task)
        obs    = result["observation"]
        done   = result["done"]

        # 2. Step Loop
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_id = get_action(client, obs, step)

            try:
                result = env_step(action_id)
                obs    = result["observation"]
                reward = float(result["reward"])
                done   = result["done"]
                info   = result.get("info", {})
                
                # If an invalid action caused an error in the environment, log it
                error = info.get("action_result") if "error" in info.get("action_result", "").lower() else None

            except Exception as env_exc:
                reward = 0.0
                done = True
                error = str(env_exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=str(action_id), reward=reward, done=done, error=error)

        # 3. Get Final Score
        state_info = env_state()
        score      = float(state_info.get("current_score", 0.0))
        
        # In our environment, score is always [0.001, 0.999]. Let's say > 0.4 is "success"
        success = score > 0.4

    except Exception as main_exc:
        print(f"[DEBUG] Episode execution failed: {main_exc}", file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    print(f"[*] Initializing baseline client connected to {API_BASE_URL} for {BENCHMARK}...", file=sys.stderr)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run tasks sequentially
    for task in TASKS:
        run_episode(client, task)


if __name__ == "__main__":
    main()