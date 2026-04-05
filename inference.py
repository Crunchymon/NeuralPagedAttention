"""
Neural PagedAttention — Baseline Inference Script
Uses an LLM (via OpenAI client) as the agent.
Emits mandatory [START] / [STEP] / [END] log format.
"""
import asyncio
import os
import sys
import textwrap
from typing import Optional

import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
BENCHMARK    = "neural-paged-attention"
MAX_STEPS    = 50          # per task episode
TEMPERATURE  = 0.0         # deterministic — required for reproducibility
MAX_TOKENS   = 10          # we only need a single integer

TASKS = ["easy", "medium", "hard"]

ACTION_DESCRIPTIONS = """
0  = Evict Largest Free cache (idle GPU only)
1  = Evict Largest VIP cache  (idle GPU only)
2  = Evict Oldest Free cache  (idle GPU only)
3  = Evict Oldest VIP cache   (idle GPU only)
4  = Swap Largest Free cache GPU→CPU
5  = Swap Largest VIP cache  GPU→CPU
6  = Swap Oldest Free cache  GPU→CPU
7  = Swap Oldest VIP cache   GPU→CPU
8  = Admit next Free user from queue
9  = Admit next VIP user from queue
10 = Reject next Free user (penalty if GPU has space)
11 = Reject next VIP user  (penalty if GPU has space)
12 = Preempt & Shred Largest Active Free request
13 = Preempt & Shred Largest Active VIP request
14 = Preempt & Swap Largest Active Free → CPU
15 = Preempt & Swap Largest Active VIP → CPU
16 = Garbage Collect (delete idle Free CPU caches > 200 ticks)
17 = Do Nothing
"""

SYSTEM_PROMPT = textwrap.dedent(f"""
You are an AI memory manager for a GPU running LLM inference.
Your job is to keep the GPU utilization high without crashing.
VIP users are 3x more valuable than Free users.
You must respond with ONLY a single integer from 0 to 17.
No explanation. No punctuation. Just the integer.

Available actions:
{ACTION_DESCRIPTIONS}
""").strip()


# ── Logging ───────────────────────────────────────────────────────────────
def log_start(task: str, model: str):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM Action Selection ──────────────────────────────────────────────────
def get_action(client: OpenAI, obs: dict, step: int) -> int:
    """
    Ask the LLM to choose an action given the current observation.
    Parses the integer from the response.
    Falls back to action 17 (Do Nothing) on any failure.
    """
    user_prompt = textwrap.dedent(f"""
    Step {step}. Current GPU state:
    - GPU utilization: {obs['gpu_utilization_pct']:.2f} (0=empty, 1=full)
    - CPU utilization: {obs['cpu_utilization_pct']:.2f}
    - Memory trend:    {obs['memory_pressure_trend']:.2f} (+rising, -falling)
    - Free queue:      {obs['free_queue_pressure']:.2f} of capacity
    - VIP queue:       {obs['vip_queue_pressure']:.2f} of capacity
    - Free SLA risk:   {obs['free_max_wait_time_pct']:.2f} (1.0 = timeout imminent)
    - VIP SLA risk:    {obs['vip_max_wait_time_pct']:.2f} (1.0 = timeout imminent)
    - Largest active:  {obs['yield_preempt_active']:.2f} of GPU
    - Largest free cache: {obs['free_size_max']:.2f}
    - Largest VIP cache:  {obs['vip_size_max']:.2f}
    - Oldest free idle:   {obs['free_age_max']:.2f}
    - Oldest VIP idle:    {obs['vip_age_max']:.2f}

    Choose one action (0-17). Reply with only the integer.
    """).strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        action_id = int(text.split()[0])
        if 0 <= action_id <= 17:
            return action_id
        return 17  # fallback
    except Exception:
        return 17  # fallback: Do Nothing


# ── Environment HTTP client ───────────────────────────────────────────────
def env_reset(task: str) -> dict:
    r = httpx.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action_id: int) -> dict:
    r = httpx.post(f"{ENV_URL}/step", json={"action_id": action_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_state() -> dict:
    r = httpx.get(f"{ENV_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()


# ── Episode Runner ────────────────────────────────────────────────────────
def run_episode(client: OpenAI, task: str) -> float:
    log_start(task=task, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    try:
        result = env_reset(task)
        obs    = result["observation"]
        done   = result["done"]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_id = get_action(client, obs, step)

            try:
                result   = env_step(action_id)
                obs      = result["observation"]
                reward   = float(result["reward"])
                done     = result["done"]
                info     = result.get("info", {})
                error_msg = info.get("action_result") if "error" in info else None
            except Exception as e:
                reward   = 0.0
                done     = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=str(action_id), reward=reward,
                     done=done, error=error_msg)

        # Get final score from state
        state  = env_state()
        score  = float(state.get("current_score", 0.0))
        success = score > 0.1

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    for task in TASKS:
        score = run_episode(client, task)
        all_scores.append(score)
        print(f"[DEBUG] Task={task} Score={score:.3f}", flush=True)

    print(f"[DEBUG] Mean score across tasks: {sum(all_scores)/len(all_scores):.3f}", flush=True)


if __name__ == "__main__":
    main()
