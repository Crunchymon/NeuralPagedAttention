import math

import server.env_components.constants as constants
from server.env_components.constants import (
    ACTIVE_GEN_BONUS,
    CRASH_PENALTY,
    FREE_MULTIPLIER,
    IDLE_PROMOTION_THRESHOLD,
    LATENCY_DECAY,
    MAX_CONCURRENT_GENERATING,
    VIP_MULTIPLIER,
)


def tick_forward(env) -> float:
    reward = 0.0
    gen_per_tick = int(env.config.get("gen_per_tick", 5))

    for req in env.free_queue + env.vip_queue:
        req.wait_ticks += 1

    for req in list(env.ledger.cpu_requests.values()):
        req.idle_ticks += 1

    # Compute-bandwidth scheduler: only MAX_CONCURRENT_GENERATING GPU-resident
    # requests actually advance this tick (FIFO by arrival_tick). The rest hold
    # cache but accrue idle_ticks; once over threshold they flip to "gpu_idle"
    # so the agent can target them with eviction / swap actions.
    gpu_residents = sorted(
        env.ledger.gpu_requests.values(),
        key=lambda r: r.arrival_tick,
    )
    generators = gpu_residents[:MAX_CONCURRENT_GENERATING]
    stalled = gpu_residents[MAX_CONCURRENT_GENERATING:]

    for req in stalled:
        req.idle_ticks += 1
        if req.idle_ticks > IDLE_PROMOTION_THRESHOLD:
            req.location = "gpu_idle"

    completed_this_tick = []
    for req in generators:
        # Resuming a previously-idle resident: reset bookkeeping so it gets
        # a fair shot at completing before it's marked idle again.
        if req.location != "gpu_active":
            req.location = "gpu_active"
            req.idle_ticks = 0
        req.generated_tokens += gen_per_tick
        reward += ACTIVE_GEN_BONUS

        new_blocks = req.current_blocks
        if new_blocks > req.blocks_allocated:
            growth = new_blocks - req.blocks_allocated
            if growth <= env.ledger.gpu_free():
                env.ledger.gpu_used += growth
                req.blocks_allocated = new_blocks
            else:
                env.crashed = True
                return CRASH_PENALTY

        if req.is_complete:
            completed_this_tick.append(req)

    for req in completed_this_tick:
        total_ticks = req.wait_ticks + req.target_gen_tokens
        multiplier = VIP_MULTIPLIER if req.tier == "vip" else FREE_MULTIPLIER
        completion_reward = multiplier * math.exp(-LATENCY_DECAY * total_ticks)
        reward += completion_reward

        fluency = req.target_gen_tokens / max(1, total_ticks)
        env._per_request_fluency.append(fluency)

        env.ledger.remove_from_gpu(req)
        req.location = "done"
        env.total_completed += 1
        env._returning_pool.append(req.request_id)

    return reward
