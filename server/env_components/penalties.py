from __future__ import annotations

import server.env_components.constants as constants
from server.env_components.constants import (
    CRASH_PENALTY,
    DEADLOCK_PENALTY,
    PAIN_95_PENALTY,
    PAIN_98_PENALTY,
    SLA_MISS_FREE,
    SLA_MISS_VIP,
)


def apply_pressure_penalty(gpu_util: float, has_idle_targets: bool) -> float:
    """Apply memory pressure penalty only when GPU is saturated and eviction targets exist."""
    if gpu_util <= 0.95:
        return 0.0
    if not has_idle_targets:
        return 0.0
    if gpu_util > 0.98:
        return PAIN_98_PENALTY
    return PAIN_95_PENALTY


def check_sla_violations(
    free_queue,
    vip_queue,
    sla_free,
    sla_vip,
) -> float:
    """Check queued requests for SLA timeout penalties."""
    penalty = 0.0

    if sla_free is not None:
        for req in free_queue:
            if req.wait_ticks >= sla_free:
                penalty += SLA_MISS_FREE

    if sla_vip is not None:
        for req in vip_queue:
            if req.wait_ticks >= sla_vip:
                penalty += SLA_MISS_VIP

    return penalty


def check_terminators(
    crashed: bool,
    gpu_used: int,
    gpu_total: int,
    free_queue_len: int,
    vip_queue_len: int,
) -> tuple[float, bool]:
    """Check for episode-ending conditions."""
    # Already flagged as crashed in a prior tick
    if crashed:
        return CRASH_PENALTY, True

    # GPU overflow — hard crash
    if gpu_used > gpu_total:
        return CRASH_PENALTY, True

    # True deadlock: both queues are full AND GPU has no room left to admit anyone
    # If GPU still has space the agent can still admit — not a deadlock
    queues_full = free_queue_len >= constants.FREE_QUEUE_MAX and vip_queue_len >= constants.VIP_QUEUE_MAX
    gpu_full = gpu_used >= gpu_total
    if queues_full and gpu_full:
        return DEADLOCK_PENALTY, True

    return 0.0, False