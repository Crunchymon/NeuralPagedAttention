"""
prompts.py — System and per-tick user prompt builders for the LLM agent.

Keeping prompts in a separate module makes it easy to iterate on prompt
engineering without touching the inference logic.
"""

from models import ACTION_MAP

# ---------------------------------------------------------------------------
# System Prompt (built once at agent init)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert GPU memory manager for a high-performance LLM inference engine \
that uses PagedAttention KV-cache scheduling.

Your job is to choose ONE action every tick to keep the system healthy:
  - Keep GPU utilization below 95% to prevent crashes
  - Prioritize VIP users over Free users
  - Avoid rejecting requests unless the system is truly saturated
  - Prefer swapping to CPU over evicting (preserves returning-user cache)
  - Garbage-collect idle CPU caches proactively to keep swap space available

AVAILABLE ACTIONS (respond with ONLY the integer ID):
""" + "\n".join(
    f"  {aid:2d} — {desc}"
    for aid, desc in ACTION_MAP.items()
) + """

RULES:
  - If GPU util < 0.85 and queues have requests → prefer admitting (8 or 9)
  - If GPU util > 0.90 → evict or swap idle caches first (0-7)
  - If GPU util > 0.95 → preempt active requests (14 or 15)
  - Never reject VIP (action 11) unless GPU util > 0.98 and VIP queue is overflowing
  - Respond with ONLY a single integer between 0 and 17. No explanation.
"""


# ---------------------------------------------------------------------------
# Per-tick user prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(obs: list[float], tick: int) -> str:
    """
    Format the current 21-dim observation into a concise, readable prompt.

    Args:
        obs:  21-dimensional observation array from KVCacheObservation.to_array()
        tick: current simulation tick number

    Returns:
        Formatted string to use as the LLM user message.
    """
    gpu_util  = obs[0]
    cpu_util  = obs[1]
    trend     = obs[2]
    free_q    = int(obs[3])
    vip_q     = int(obs[4])
    total_q   = int(obs[5])
    free_wait = obs[6]
    vip_wait  = obs[7]
    yield_pre = obs[8]
    free_age  = obs[14]
    vip_age   = obs[17]

    # Human-readable pressure labels
    def _label(v: float, hi=0.90, mid=0.75) -> str:
        if v >= hi:   return "CRITICAL"
        if v >= mid:  return "HIGH"
        if v >= 0.50: return "MODERATE"
        return "OK"

    return (
        f"Tick {tick} — Current State:\n"
        f"  GPU utilization : {gpu_util:.2f}  [{_label(gpu_util)}]\n"
        f"  CPU utilization : {cpu_util:.2f}  [{_label(cpu_util, 0.85, 0.60)}]\n"
        f"  Pressure trend  : {trend:+.2f}  ({'rising' if trend > 0.05 else 'falling' if trend < -0.05 else 'stable'})\n"
        f"  Free queue      : {free_q} requests waiting  (SLA risk {free_wait:.2f})\n"
        f"  VIP queue       : {vip_q} requests waiting   (SLA risk {vip_wait:.2f})\n"
        f"  Total queued    : {total_q}\n"
        f"  Oldest free age : {free_age:.2f}  (1.0 = very stale)\n"
        f"  Oldest VIP age  : {vip_age:.2f}\n"
        f"  Preempt yield   : {yield_pre:.2f}\n"
        f"\nRespond with ONLY a single integer 0-17:"
    )
