import math

from server.env_components.constants import (
    ACTIVE_GEN_BONUS,
    CRASH_PENALTY,
    FREE_MULTIPLIER,
    LATENCY_DECAY,
    VIP_MULTIPLIER,
)


def tick_forward(env) -> float:
    reward = 0.0

    for req in env.free_queue + env.vip_queue:
        req.wait_ticks += 1

    for req in list(env.ledger.gpu_requests.values()):
        if req.is_idle_gpu:
            req.idle_ticks += 1

    for req in list(env.ledger.cpu_requests.values()):
        req.idle_ticks += 1

    completed_this_tick = []
    for req in list(env.ledger.gpu_requests.values()):
        if req.is_active:
            req.generated_tokens += 5
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
