from __future__ import annotations

import numpy as np

from models import KVCacheObservation


def build_observation(
    free_queue,
    vip_queue,
    ledger,
    gpu_history: list[float],
    gpu_total_blocks: int,
    free_queue_max: int,
    vip_queue_max: int,
    sla_free: int,
    sla_vip: int,
) -> KVCacheObservation:
    """Construct the normalized 20-dimensional observation."""
    if len(gpu_history) >= 2:
        trend = gpu_history[-1] - gpu_history[0]
        trend = max(-1.0, min(1.0, trend))
    else:
        trend = 0.0

    total_free_req = float(len(free_queue))
    total_vip_req = float(len(vip_queue))
    total_req = total_free_req + total_vip_req

    oldest_free_wait = max((r.wait_ticks for r in free_queue), default=0)
    oldest_vip_wait = max((r.wait_ticks for r in vip_queue), default=0)
    free_wait_pct = min(1.0, oldest_free_wait / sla_free)
    vip_wait_pct = min(1.0, oldest_vip_wait / sla_vip)

    active_reqs = ledger.active_gpu_requests()
    largest_active = max((r.current_blocks for r in active_reqs), default=0)
    yield_preempt = min(1.0, largest_active / gpu_total_blocks)

    def _stats(reqs) -> tuple[float, float, float]:
        if not reqs:
            return 0.0, 0.0, 0.0
        sizes = [r.current_blocks / gpu_total_blocks for r in reqs]
        return (
            min(1.0, max(sizes)),
            min(1.0, float(np.mean(sizes))),
            min(1.0, float(np.std(sizes))),
        )

    def _age_stats(reqs) -> tuple[float, float, float]:
        if not reqs:
            return 0.0, 0.0, 0.0
        ages = [min(1.0, r.idle_ticks / 500) for r in reqs]
        return (
            min(1.0, max(ages)),
            min(1.0, float(np.mean(ages))),
            min(1.0, float(np.std(ages))),
        )

    free_idle = ledger.idle_gpu_requests("free")
    vip_idle = ledger.idle_gpu_requests("vip")

    fs_max, fs_mean, fs_std = _stats(free_idle)
    vs_max, vs_mean, vs_std = _stats(vip_idle)
    fa_max, fa_mean, fa_std = _age_stats(free_idle)
    va_max, va_mean, va_std = _age_stats(vip_idle)

    return KVCacheObservation(
        gpu_utilization_pct=min(1.0, ledger.gpu_utilization()),
        cpu_utilization_pct=min(1.0, ledger.cpu_utilization()),
        memory_pressure_trend=trend,
        total_free_req=total_free_req,
        total_vip_req=total_vip_req,
        total_req=total_req,
        free_max_wait_time_pct=free_wait_pct,
        vip_max_wait_time_pct=vip_wait_pct,
        yield_preempt_active=yield_preempt,
        free_size_max=fs_max,
        free_size_mean=fs_mean,
        free_size_std_dev=fs_std,
        vip_size_max=vs_max,
        vip_size_mean=vs_mean,
        vip_size_std_dev=vs_std,
        free_age_max=fa_max,
        free_age_mean=fa_mean,
        free_age_std_dev=fa_std,
        vip_age_max=va_max,
        vip_age_mean=va_mean,
        vip_age_std_dev=va_std,
    )