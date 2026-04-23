from __future__ import annotations

import numpy as np


def compute_final_score(
    task: str,
    total_completed: int,
    total_arrived: int,
    per_request_fluency: list[float],
    total_cache_hits: int,
    total_returning_arrived: int,
    total_swaps: int,
    total_actions: int,
) -> float:
    """
    Compute the final episodic score, always STRICTLY in (0.001, 0.999).

    The task-specific weights match the current grader behavior exactly.
    """
    v1 = total_completed / max(1, total_arrived)

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
    v3 = min(1.0, max(0.0, cache_hit_rate - swap_rate))

    if task == "easy":
        raw = 0.80 * v1 + 0.20 * v2
    elif task == "medium":
        raw = 0.50 * v1 + 0.30 * v2 + 0.20 * v3
    else:
        raw = 0.40 * v1 + 0.35 * v2 + 0.25 * v3

    return 0.001 + 0.998 * min(1.0, max(0.0, raw))