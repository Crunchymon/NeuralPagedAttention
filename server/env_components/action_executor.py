from server.env_components.constants import (
    ADMIT_BONUS,
    DO_NOTHING_TAX,
    EVICT_BONUS,
    GC_BONUS,
    GC_IDLE_THRESHOLD,
    INVALID_ACTION_PENALTY,
    PREEMPT_SHRED_FREE,
    PREEMPT_SHRED_VIP,
    PREEMPT_SWAP_FREE,
    PREEMPT_SWAP_VIP,
    REJECT_NECESSARY_FREE,
    REJECT_NECESSARY_VIP,
    REJECT_UNNECESSARY_FREE,
    REJECT_UNNECESSARY_VIP,
    SWAP_TAX_FREE,
    SWAP_TAX_VIP,
)


def _pick(reqs: list, by: str):
    if by == "size":
        return max(reqs, key=lambda r: r.current_blocks)
    return max(reqs, key=lambda r: r.idle_ticks)


def _silent_janitor(env, needed: int):
    free_cpu = [r for r in env.ledger.cpu_requests.values() if r.tier == "free"]
    free_cpu.sort(key=lambda r: r.idle_ticks, reverse=True)
    freed = 0
    for req in free_cpu:
        if freed >= needed:
            break
        env.ledger.remove_from_cpu(req)
        req.location = "evicted"
        freed += req.blocks_allocated


def _evict_idle(env, tier: str, by: str) -> float:
    candidates = env.ledger.idle_gpu_requests(tier)
    if not candidates:
        env._last_action_result = f"evict_{tier}_{by}_no_target"
        return INVALID_ACTION_PENALTY
    target = _pick(candidates, by)
    env.ledger.remove_from_gpu(target)
    target.location = "evicted"
    env._last_action_result = f"evicted_{tier}_{by}_{target.request_id}"
    return EVICT_BONUS


def _swap_to_cpu_idle(env, tier: str, by: str) -> float:
    candidates = env.ledger.idle_gpu_requests(tier)
    if not candidates:
        env._last_action_result = f"swap_{tier}_{by}_no_target"
        return INVALID_ACTION_PENALTY
    target = _pick(candidates, by)
    env.ledger.remove_from_gpu(target)

    if target.current_blocks > env.ledger.cpu_free():
        _silent_janitor(env, needed=target.current_blocks)

    env.ledger.place_on_cpu(target)
    env.total_swaps += 1
    tax = SWAP_TAX_FREE if tier == "free" else SWAP_TAX_VIP
    env._last_action_result = f"swapped_{tier}_{by}_{target.request_id}_to_cpu"
    return tax


def _admit_next(env, tier: str) -> float:
    queue = env.free_queue if tier == "free" else env.vip_queue
    if not queue:
        env._last_action_result = f"admit_{tier}_empty_queue"
        return INVALID_ACTION_PENALTY

    req = queue[0]
    blocks_needed = req.current_blocks

    if blocks_needed > env.ledger.gpu_free():
        env._last_action_result = f"admit_{tier}_no_gpu_space"
        return INVALID_ACTION_PENALTY

    queue.pop(0)

    if req.is_returning:
        env.total_returning_arrived += 1
        if req.request_id in env.ledger.cpu_requests:
            env.total_cache_hits += 1
            cpu_req = env.ledger.cpu_requests[req.request_id]
            env.ledger.remove_from_cpu(cpu_req)
            req.generated_tokens = cpu_req.generated_tokens
        elif req.request_id in env.ledger.gpu_requests:
            env.total_cache_hits += 1
            req = env.ledger.gpu_requests[req.request_id]
            req.location = "gpu_active"
            env._last_action_result = f"admitted_{tier}_cache_hit_gpu"
            return ADMIT_BONUS

    env.ledger.place_on_gpu(req, location="gpu_active")
    env._last_action_result = f"admitted_{tier}_{req.request_id}"
    return ADMIT_BONUS


def admit_batch(env, tier: str, pct: float) -> float:
    """Admit up to pct% of the queue, limited by available GPU blocks.
    Returns cumulative reward from all individual admissions."""
    pct = max(0.0, min(1.0, pct))
    queue = env.free_queue if tier == "free" else env.vip_queue
    if not queue:
        return INVALID_ACTION_PENALTY

    n_to_admit = max(1, round(len(queue) * pct))
    total_reward = 0.0
    admitted = 0

    for _ in range(n_to_admit):
        queue = env.free_queue if tier == "free" else env.vip_queue
        if not queue:
            break
        req = queue[0]
        if req.current_blocks > env.ledger.gpu_free():
            break  # No more GPU space
        total_reward += _admit_next(env, tier)
        admitted += 1

    if admitted == 0:
        env._last_action_result = f"admit_batch_{tier}_no_gpu_space"
        return INVALID_ACTION_PENALTY

    env._last_action_result = f"admit_batch_{tier}_{admitted}_of_{n_to_admit}"
    return total_reward


def _reject_next(env, tier: str) -> float:
    queue = env.free_queue if tier == "free" else env.vip_queue
    if not queue:
        env._last_action_result = f"reject_{tier}_empty_queue"
        return INVALID_ACTION_PENALTY

    req = queue.pop(0)
    env.total_rejected += 1

    gpu_has_space = env.ledger.gpu_free() >= req.current_blocks
    if gpu_has_space:
        penalty = REJECT_UNNECESSARY_FREE if tier == "free" else REJECT_UNNECESSARY_VIP
        env._last_action_result = f"rejected_{tier}_unnecessary"
    else:
        penalty = REJECT_NECESSARY_FREE if tier == "free" else REJECT_NECESSARY_VIP
        env._last_action_result = f"rejected_{tier}_necessary"

    return penalty


def _preempt_shred(env, tier: str) -> float:
    candidates = env.ledger.active_gpu_requests(tier)
    if not candidates:
        env._last_action_result = f"preempt_shred_{tier}_no_target"
        return INVALID_ACTION_PENALTY
    target = _pick(candidates, by="size")
    env.ledger.remove_from_gpu(target)
    target.location = "evicted"
    penalty = PREEMPT_SHRED_FREE if tier == "free" else PREEMPT_SHRED_VIP
    env._last_action_result = f"preempted_shred_{tier}_{target.request_id}"
    return penalty


def _preempt_swap(env, tier: str) -> float:
    candidates = env.ledger.active_gpu_requests(tier)
    if not candidates:
        env._last_action_result = f"preempt_swap_{tier}_no_target"
        return INVALID_ACTION_PENALTY
    target = _pick(candidates, by="size")
    env.ledger.remove_from_gpu(target)
    target.location = "queue"

    if target.current_blocks > env.ledger.cpu_free():
        _silent_janitor(env, needed=target.current_blocks)

    env.ledger.place_on_cpu(target)
    if tier == "vip":
        env.vip_queue.insert(0, target)
    else:
        env.free_queue.insert(0, target)

    env.total_swaps += 1
    penalty = PREEMPT_SWAP_FREE if tier == "free" else PREEMPT_SWAP_VIP
    env._last_action_result = f"preempted_swap_{tier}_{target.request_id}_to_cpu"
    return penalty


def _garbage_collect(env) -> float:
    to_remove = [
        r for r in env.ledger.cpu_requests.values()
        if r.tier == "free" and r.idle_ticks > GC_IDLE_THRESHOLD
    ]
    if not to_remove:
        env._last_action_result = "gc_removed_0_caches"
        return INVALID_ACTION_PENALTY

    for req in to_remove:
        env.ledger.remove_from_cpu(req)
        req.location = "evicted"
    env._last_action_result = f"gc_removed_{len(to_remove)}_caches"
    return GC_BONUS


def execute_action(env, action_id: int) -> float:
    handlers = {
        0: lambda: _evict_idle(env, "free", by="size"),
        1: lambda: _evict_idle(env, "vip", by="size"),
        2: lambda: _evict_idle(env, "free", by="age"),
        3: lambda: _evict_idle(env, "vip", by="age"),
        4: lambda: _swap_to_cpu_idle(env, "free", by="size"),
        5: lambda: _swap_to_cpu_idle(env, "vip", by="size"),
        6: lambda: _swap_to_cpu_idle(env, "free", by="age"),
        7: lambda: _swap_to_cpu_idle(env, "vip", by="age"),
        8: lambda: _admit_next(env, "free"),
        9: lambda: _admit_next(env, "vip"),
        10: lambda: _reject_next(env, "free"),
        11: lambda: _reject_next(env, "vip"),
        12: lambda: _preempt_shred(env, "free"),
        13: lambda: _preempt_shred(env, "vip"),
        14: lambda: _preempt_swap(env, "free"),
        15: lambda: _preempt_swap(env, "vip"),
        16: lambda: _garbage_collect(env),
        17: lambda: DO_NOTHING_TAX,
    }
    fn = handlers.get(action_id, lambda: 0.0)
    return fn()
