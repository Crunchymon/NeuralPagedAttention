from __future__ import annotations

import numpy as np

from server.env_components.constants import PHASE_CONFIGS


def compute_final_score(
    task: str,
    total_completed: int,
    total_arrived: int,
    per_request_fluency: list[float],
    total_cache_hits: int,
    total_returning_arrived: int,
    total_swaps: int,
    total_actions: int,
    crashed: bool = False,
) -> float:
    """
    Compute the final episodic score, always STRICTLY in (0.001, 0.999).

    v1 is normalized against a per-phase reachable-completion ceiling so that
    the structurally infeasible portions of v1 (i.e., requests no policy could
    have served given the GPU/throughput/traffic budget) don't drag the score.
    v3 is mapped from the symmetric range [-1, 1] to [0, 1] so swap-spam
    actually reduces v3 instead of being free.
    Crashing halves the final raw score before bounding.
    """
    raw_completion = total_completed / max(1, total_arrived)
    ceiling = float(PHASE_CONFIGS.get(task, {}).get("v1_ceiling", 1.0))
    ceiling = max(1e-3, min(1.0, ceiling))
    v1 = min(1.0, raw_completion / ceiling)

    if per_request_fluency:
        v2 = float(np.mean(per_request_fluency))
    else:
        v2 = 0.0
    v2 = min(1.0, max(0.0, v2))

    cache_hit_rate = (
        total_cache_hits / max(1, total_returning_arrived)
        if total_returning_arrived > 0 else 0.0
    )
    swap_rate = total_swaps / max(1, total_actions)
    v3_signed = max(-1.0, min(1.0, cache_hit_rate - swap_rate))
    v3 = (v3_signed + 1.0) / 2.0

    if task == "easy":
        raw = 0.80 * v1 + 0.20 * v2
    elif task == "medium":
        raw = 0.50 * v1 + 0.30 * v2 + 0.20 * v3
    else:
        raw = 0.40 * v1 + 0.35 * v2 + 0.25 * v3

    if crashed:
        raw *= 0.5

    return 0.001 + 0.998 * min(1.0, max(0.0, raw))
