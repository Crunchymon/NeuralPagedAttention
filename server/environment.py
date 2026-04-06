import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Hardware ──────────────────────────────────────────────────────────────
GPU_TOTAL_BLOCKS   = 10_000   # 160,000 tokens
CPU_TOTAL_BLOCKS   = 50_00   # 800,000 tokens
TOKENS_PER_BLOCK   = 16

# ── Traffic ───────────────────────────────────────────────────────────────
CHATTER_RATIO      = 0.80
POWER_USER_RATIO   = 0.20
VIP_FREE_DEFAULT   = 0.05     # overridden per phase
RETURNING_RATIO    = 0.30

# Chatter token distributions
CHATTER_PROMPT_MU    = 100;  CHATTER_PROMPT_SIGMA    = 50
CHATTER_GEN_MU       = 250;  CHATTER_GEN_SIGMA       = 100

# Power user token distributions
POWER_PROMPT_MU      = 3500; POWER_PROMPT_SIGMA      = 1200
POWER_GEN_MU         = 800;  POWER_GEN_SIGMA         = 300

# ── Queue limits ──────────────────────────────────────────────────────────
FREE_QUEUE_MAX     = 100
VIP_QUEUE_MAX      = 50

# ── Reward constants ──────────────────────────────────────────────────────
FREE_MULTIPLIER    = 5.0
VIP_MULTIPLIER     = 15.0
LATENCY_DECAY      = 0.05

# Pain receptor thresholds
PAIN_95_PENALTY    = -0.1
PAIN_98_PENALTY    = -0.4

# Swap tax
SWAP_TAX_FREE      = -0.2
SWAP_TAX_VIP       = -0.5

# Preempt costs
PREEMPT_SHRED_FREE = -1.0
PREEMPT_SHRED_VIP  = -3.0
PREEMPT_SWAP_FREE  = -0.6   # 40% discount vs shred
PREEMPT_SWAP_VIP   = -1.8

# Reject costs
REJECT_UNNECESSARY_FREE = -5.0
REJECT_UNNECESSARY_VIP  = -10.0
REJECT_NECESSARY_FREE   = -0.5
REJECT_NECESSARY_VIP    = -2.0

# Terminators
SLA_MISS_FREE      = -10.0
SLA_MISS_VIP       = -30.0
DEADLOCK_PENALTY   = -20.0
CRASH_PENALTY      = -100.0

# DQN Reward Shaping
INVALID_ACTION_PENALTY = -1.0
DO_NOTHING_TAX         = -0.01
ADMIT_BONUS            = 0.1
EVICT_BONUS            = 0.05
GC_BONUS               = 0.2
ACTIVE_GEN_BONUS       = 0.02

# Garbage collect idle threshold (ticks)
GC_IDLE_THRESHOLD  = 200

# ── Phase configs ─────────────────────────────────────────────────────────
PHASE_CONFIGS = {
    "easy": {
        "max_ticks":       2000,
        "vip_ratio":       0.02,
        "sla_free":        None,   # None = disabled
        "sla_vip":         None,
        "traffic_fn":      "flat",  # 1 request/tick constant
        "power_user_pct":  0.0,    # no power users in easy
    },
    "medium": {
        "max_ticks":       5000,
        "vip_ratio":       0.05,
        "sla_free":        100,
        "sla_vip":         50,
        "traffic_fn":      "wave",
        "power_user_pct":  0.20,
    },
    "hard": {
        "max_ticks":       10000,
        "vip_ratio":       0.10,
        "sla_free":        50,
        "sla_vip":         25,
        "traffic_fn":      "spike",
        "power_user_pct":  0.35,
    },
}


# ── Data Structures ───────────────────────────────────────────────────────

@dataclass
class Request:
    """A single LLM inference request."""
    request_id:       str
    tier:             str          # "free" | "vip"
    user_type:        str          # "chatter" | "power"
    is_returning:     bool
    prompt_tokens:    int
    target_gen_tokens: int
    generated_tokens: int = 0
    wait_ticks:       int = 0
    location:         str = "queue"  # "queue" | "gpu_active" | "gpu_idle" | "cpu" | "done"
    blocks_allocated: int = 0
    idle_ticks:       int = 0        # ticks spent idle on gpu or cpu (not actively generating)
    arrival_tick:     int = 0

    @property
    def total_blocks_needed(self) -> int:
        total_tokens = self.prompt_tokens + self.target_gen_tokens
        return math.ceil(total_tokens / TOKENS_PER_BLOCK)

    @property
    def current_blocks(self) -> int:
        used_tokens = self.prompt_tokens + self.generated_tokens
        return math.ceil(used_tokens / TOKENS_PER_BLOCK)

    @property
    def is_complete(self) -> bool:
        return self.generated_tokens >= self.target_gen_tokens

    @property
    def is_active(self) -> bool:
        return self.location == "gpu_active"

    @property
    def is_idle_gpu(self) -> bool:
        return self.location == "gpu_idle"

    @property
    def is_on_cpu(self) -> bool:
        return self.location == "cpu"


class MemoryLedger:
    """
    Tracks GPU and CPU block usage.
    Uses simple lists — O(n) operations are acceptable at this scale.
    The environment spec calls for O(1) heaps but for correctness and
    simplicity use sorted lists. Optimise only if performance requires it.
    """
    def __init__(self):
        self.gpu_used: int = 0
        self.cpu_used: int = 0
        self.gpu_requests: dict[str, Request] = {}  # id -> Request on GPU
        self.cpu_requests: dict[str, Request] = {}  # id -> Request on CPU

    def gpu_free(self) -> int:
        return GPU_TOTAL_BLOCKS - self.gpu_used

    def cpu_free(self) -> int:
        return CPU_TOTAL_BLOCKS - self.cpu_used

    def gpu_utilization(self) -> float:
        return self.gpu_used / GPU_TOTAL_BLOCKS

    def cpu_utilization(self) -> float:
        return self.cpu_used / CPU_TOTAL_BLOCKS

    def place_on_gpu(self, req: Request, location: str = "gpu_active") -> bool:
        """Returns False if not enough GPU space."""
        blocks = req.current_blocks
        if blocks > self.gpu_free():
            return False
        req.location = location
        req.blocks_allocated = blocks
        self.gpu_used += blocks
        self.gpu_requests[req.request_id] = req
        return True

    def remove_from_gpu(self, req: Request):
        self.gpu_used -= req.blocks_allocated
        self.gpu_requests.pop(req.request_id, None)
        req.blocks_allocated = 0

    def place_on_cpu(self, req: Request) -> bool:
        """Returns False if not enough CPU space."""
        blocks = req.current_blocks
        if blocks > self.cpu_free():
            return False
        req.location = "cpu"
        req.blocks_allocated = blocks
        self.cpu_used += blocks
        self.cpu_requests[req.request_id] = req
        return True

    def remove_from_cpu(self, req: Request):
        self.cpu_used -= req.blocks_allocated
        self.cpu_requests.pop(req.request_id, None)
        req.blocks_allocated = 0

    def idle_gpu_requests(self, tier: Optional[str] = None) -> list[Request]:
        """Return gpu_idle requests, optionally filtered by tier."""
        reqs = [r for r in self.gpu_requests.values() if r.is_idle_gpu]
        if tier:
            reqs = [r for r in reqs if r.tier == tier]
        return reqs

    def active_gpu_requests(self, tier: Optional[str] = None) -> list[Request]:
        reqs = [r for r in self.gpu_requests.values() if r.is_active]
        if tier:
            reqs = [r for r in reqs if r.tier == tier]
        return reqs

    def cpu_requests_by_tier(self, tier: str) -> list[Request]:
        return [r for r in self.cpu_requests.values() if r.tier == tier]


# ── Traffic Generator ─────────────────────────────────────────────────────

def _traffic_flat(t: int) -> int:
    """Easy phase: constant 1 request per tick."""
    return 1

def _traffic_wave(t: int) -> int:
    """Medium phase: day/night wave, average ~2/tick."""
    rate = 2 + 0.005 * t + 4 * math.sin(2 * math.pi * t / 1000) + 2 * math.sin(2 * math.pi * t / 150)
    return max(0, round(rate))

def _traffic_spike(t: int) -> int:
    """Hard phase: violent viral spikes."""
    rate = 4 + 0.005 * t + 8 * math.sin(2 * math.pi * t / 1000) + 4 * math.sin(2 * math.pi * t / 150)
    return max(0, round(rate))

TRAFFIC_FNS = {
    "flat":  _traffic_flat,
    "wave":  _traffic_wave,
    "spike": _traffic_spike,
}


def generate_request(
    tick: int,
    vip_ratio: float,
    power_user_pct: float,
    returning_pool: list[str],   # list of request IDs eligible to return
    rng: random.Random,
) -> Request:
    """
    Generate one new Request object.

    In easy phase, power_user_pct=0.0 so all users are chatters.
    30% of all requests are returning users (if returning_pool is non-empty).
    """
    tier = "vip" if rng.random() < vip_ratio else "free"

    is_power = rng.random() < power_user_pct
    user_type = "power" if is_power else "chatter"

    is_returning = (
        len(returning_pool) > 0 and rng.random() < RETURNING_RATIO
    )

    if user_type == "chatter":
        prompt = max(16, int(rng.gauss(CHATTER_PROMPT_MU, CHATTER_PROMPT_SIGMA)))
        gen    = max(16, int(rng.gauss(CHATTER_GEN_MU,    CHATTER_GEN_SIGMA)))
    else:
        prompt = max(16, int(rng.gauss(POWER_PROMPT_MU, POWER_PROMPT_SIGMA)))
        gen    = max(16, int(rng.gauss(POWER_GEN_MU,    POWER_GEN_SIGMA)))

    # Cap total tokens to fit within GPU in extreme cases
    # Max is 80% of GPU capacity to leave room for other requests
    max_tokens = int(GPU_TOTAL_BLOCKS * 0.8 * TOKENS_PER_BLOCK)
    if prompt + gen > max_tokens:
        gen = max(16, max_tokens - prompt)

    return Request(
        request_id=str(uuid.uuid4())[:8],
        tier=tier,
        user_type=user_type,
        is_returning=is_returning,
        prompt_tokens=prompt,
        target_gen_tokens=gen,
        arrival_tick=tick,
    )


# ── Main Environment Class ────────────────────────────────────────────────

class KVCacheEnvironment:
    """
    The core simulation. Implements reset(), step(), state().
    
    State machine per episode:
      reset(task) → initialises everything → returns initial observation
      step(action_id) → executes action, advances tick, returns (obs, reward, done, info)
      state() → returns KVCacheState metadata
    """

    def __init__(self):
        self.task: str = "easy"
        self.config: dict = PHASE_CONFIGS[self.task]
        self.tick: int = 0
        self.episode_id: str = ""
        self.rng: random.Random = random.Random()

        # Queues
        self.free_queue: list[Request] = []
        self.vip_queue:  list[Request] = []

        # Memory
        self.ledger: MemoryLedger = MemoryLedger()

        # Episode tracking
        self.total_arrived:   int = 0
        self.total_completed: int = 0
        self.total_rejected:  int = 0
        self.total_swaps:     int = 0
        self.total_actions:   int = 0
        self.total_cache_hits: int = 0
        self.total_returning_arrived: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.crashed: bool = False

        # For v2 score computation
        self._per_request_fluency: list[float] = []

        # GPU history for pressure trend (last 5 ticks)
        self._gpu_history: list[float] = []

        # Pool of completed request IDs eligible to return
        self._returning_pool: list[str] = []

        # Track last action for info dict
        self._last_action_result: str = "none"

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, task: str = "easy") -> "KVCacheObservation":
        """
        Reset the environment for a new episode.
        task must be one of: "easy", "medium", "hard"
        """
        assert task in PHASE_CONFIGS, f"Unknown task: {task}"
        self.task    = task
        self.config  = PHASE_CONFIGS[task]
        self.tick    = 0
        self.episode_id = str(uuid.uuid4())[:12]
        self.rng     = random.Random(42)  # deterministic seed for reproducibility

        self.free_queue = []
        self.vip_queue  = []
        self.ledger     = MemoryLedger()

        self.total_arrived   = 0
        self.total_completed = 0
        self.total_rejected  = 0
        self.total_swaps     = 0
        self.total_actions   = 0
        self.total_cache_hits = 0
        self.total_returning_arrived = 0
        self.cumulative_reward = 0.0
        self.done    = False
        self.crashed = False

        self._per_request_fluency = []
        self._gpu_history         = []
        self._returning_pool      = []
        self._last_action_result  = "reset"

        # Spawn initial traffic
        self._spawn_traffic()

        return self._build_observation()

    def step(self, action_id: int) -> tuple:
        """
        Execute one action and advance the simulation by one tick.
        Returns: (KVCacheObservation, float reward, bool done, dict info)
        """
        if self.done:
            return self._build_observation(), 0.0, True, {"error": "episode_done"}

        reward = 0.0
        self.total_actions += 1

        # ── 1. Execute the chosen action ──────────────────────────────────
        reward += self._execute_action(action_id)

        # ── 2. Advance simulation by one tick ────────────────────────────
        reward += self._tick_forward()

        # ── 3. Spawn new traffic ──────────────────────────────────────────
        self._spawn_traffic()

        # ── 4. Apply memory pressure penalty ─────────────────────────────
        reward += self._apply_pressure_penalty()

        # ── 5. Check SLA violations ───────────────────────────────────────
        sla_penalty = self._check_sla_violations()
        reward += sla_penalty

        # ── 6. Check terminators ──────────────────────────────────────────
        terminator_penalty, terminated = self._check_terminators()
        reward += terminator_penalty

        # ── 7. Update state ───────────────────────────────────────────────
        self.cumulative_reward += reward
        self._gpu_history.append(self.ledger.gpu_utilization())
        if len(self._gpu_history) > 5:
            self._gpu_history.pop(0)

        if terminated or self.tick >= self.config["max_ticks"]:
            self.done = True

        self.tick += 1

        info = {
            "action": action_id,
            "action_result": self._last_action_result,
            "tick": self.tick,
            "gpu_util": self.ledger.gpu_utilization(),
            "sla_penalty": sla_penalty,
        }

        return self._build_observation(), reward, self.done, info

    def state(self) -> "KVCacheState":
        from models import KVCacheState
        return KVCacheState(
            episode_id=self.episode_id,
            task=self.task,
            tick=self.tick,
            max_ticks=self.config["max_ticks"],
            total_arrived=self.total_arrived,
            total_completed=self.total_completed,
            total_rejected=self.total_rejected,
            total_crashed=self.crashed,
            current_score=self._compute_final_score(),
            cumulative_reward=self.cumulative_reward,
        )

    # ──────────────────────────────────────────────────────────────────────
    # ACTION EXECUTION
    # ──────────────────────────────────────────────────────────────────────

    def _execute_action(self, action_id: int) -> float:
        """
        Dispatch action to correct handler. Returns immediate reward.
        
        Action map:
          0  Evict Largest Free (idle GPU only)
          1  Evict Largest VIP  (idle GPU only)
          2  Evict Oldest Free  (idle GPU only)
          3  Evict Oldest VIP   (idle GPU only)
          4  Swap Largest Free GPU→CPU
          5  Swap Largest VIP  GPU→CPU
          6  Swap Oldest Free  GPU→CPU
          7  Swap Oldest VIP   GPU→CPU
          8  Admit Next Free from queue
          9  Admit Next VIP from queue
          10 Reject Next Free
          11 Reject Next VIP
          12 Preempt Shred Largest Active Free
          13 Preempt Shred Largest Active VIP
          14 Preempt Swap Largest Active Free → CPU
          15 Preempt Swap Largest Active VIP  → CPU
          16 Garbage Collect (delete Free CPU caches idle > 200 ticks)
          17 Do Nothing
        """
        handlers = {
            0:  lambda: self._evict_idle("free", by="size"),
            1:  lambda: self._evict_idle("vip",  by="size"),
            2:  lambda: self._evict_idle("free", by="age"),
            3:  lambda: self._evict_idle("vip",  by="age"),
            4:  lambda: self._swap_to_cpu_idle("free", by="size"),
            5:  lambda: self._swap_to_cpu_idle("vip",  by="size"),
            6:  lambda: self._swap_to_cpu_idle("free", by="age"),
            7:  lambda: self._swap_to_cpu_idle("vip",  by="age"),
            8:  lambda: self._admit_next("free"),
            9:  lambda: self._admit_next("vip"),
            10: lambda: self._reject_next("free"),
            11: lambda: self._reject_next("vip"),
            12: lambda: self._preempt_shred("free"),
            13: lambda: self._preempt_shred("vip"),
            14: lambda: self._preempt_swap("free"),
            15: lambda: self._preempt_swap("vip"),
            16: lambda: self._garbage_collect(),
            17: lambda: DO_NOTHING_TAX,  # Do Nothing
        }
        fn = handlers.get(action_id, lambda: 0.0)
        return fn()

    def _evict_idle(self, tier: str, by: str) -> float:
        """Delete an idle GPU cache. Returns penalty if failed."""
        candidates = self.ledger.idle_gpu_requests(tier)
        if not candidates:
            self._last_action_result = f"evict_{tier}_{by}_no_target"
            return INVALID_ACTION_PENALTY
        target = self._pick(candidates, by)
        self.ledger.remove_from_gpu(target)
        target.location = "evicted"
        self._last_action_result = f"evicted_{tier}_{by}_{target.request_id}"
        return EVICT_BONUS

    def _swap_to_cpu_idle(self, tier: str, by: str) -> float:
        """Move idle GPU cache to CPU. Returns swap tax."""
        candidates = self.ledger.idle_gpu_requests(tier)
        if not candidates:
            self._last_action_result = f"swap_{tier}_{by}_no_target"
            return INVALID_ACTION_PENALTY
        target = self._pick(candidates, by)
        self.ledger.remove_from_gpu(target)

        # CPU overflow: silent janitor fires if CPU is full
        if target.current_blocks > self.ledger.cpu_free():
            self._silent_janitor(needed=target.current_blocks)

        self.ledger.place_on_cpu(target)
        self.total_swaps += 1
        tax = SWAP_TAX_FREE if tier == "free" else SWAP_TAX_VIP
        self._last_action_result = f"swapped_{tier}_{by}_{target.request_id}_to_cpu"
        return tax

    def _admit_next(self, tier: str) -> float:
        """Admit next request from queue to GPU."""
        queue = self.free_queue if tier == "free" else self.vip_queue
        if not queue:
            self._last_action_result = f"admit_{tier}_empty_queue"
            return INVALID_ACTION_PENALTY

        req = queue[0]
        blocks_needed = req.current_blocks

        if blocks_needed > self.ledger.gpu_free():
            self._last_action_result = f"admit_{tier}_no_gpu_space"
            return INVALID_ACTION_PENALTY

        queue.pop(0)

        # Cache hit detection for returning users
        if req.is_returning:
            self.total_returning_arrived += 1
            # Check if their cache is on CPU (best case) or still on GPU idle
            if req.request_id in self.ledger.cpu_requests:
                self.total_cache_hits += 1
                cpu_req = self.ledger.cpu_requests[req.request_id]
                self.ledger.remove_from_cpu(cpu_req)
                req.generated_tokens = cpu_req.generated_tokens
            elif req.request_id in self.ledger.gpu_requests:
                self.total_cache_hits += 1
                # Already on GPU idle, just activate it
                req = self.ledger.gpu_requests[req.request_id]
                req.location = "gpu_active"
                self._last_action_result = f"admitted_{tier}_cache_hit_gpu"
                return ADMIT_BONUS

        self.ledger.place_on_gpu(req, location="gpu_active")
        self._last_action_result = f"admitted_{tier}_{req.request_id}"
        return ADMIT_BONUS

    def _reject_next(self, tier: str) -> float:
        """Drop next queued request. Penalty depends on whether GPU has space."""
        queue = self.free_queue if tier == "free" else self.vip_queue
        if not queue:
            self._last_action_result = f"reject_{tier}_empty_queue"
            return INVALID_ACTION_PENALTY

        req = queue.pop(0)
        self.total_rejected += 1

        gpu_has_space = self.ledger.gpu_free() >= req.current_blocks
        if gpu_has_space:
            penalty = REJECT_UNNECESSARY_FREE if tier == "free" else REJECT_UNNECESSARY_VIP
            self._last_action_result = f"rejected_{tier}_unnecessary"
        else:
            penalty = REJECT_NECESSARY_FREE if tier == "free" else REJECT_NECESSARY_VIP
            self._last_action_result = f"rejected_{tier}_necessary"

        return penalty

    def _preempt_shred(self, tier: str) -> float:
        """Interrupt and permanently delete an active GPU request."""
        candidates = self.ledger.active_gpu_requests(tier)
        if not candidates:
            self._last_action_result = f"preempt_shred_{tier}_no_target"
            return INVALID_ACTION_PENALTY
        target = self._pick(candidates, by="size")
        self.ledger.remove_from_gpu(target)
        target.location = "evicted"
        penalty = PREEMPT_SHRED_FREE if tier == "free" else PREEMPT_SHRED_VIP
        self._last_action_result = f"preempted_shred_{tier}_{target.request_id}"
        return penalty

    def _preempt_swap(self, tier: str) -> float:
        """Interrupt active GPU request and move to CPU."""
        candidates = self.ledger.active_gpu_requests(tier)
        if not candidates:
            self._last_action_result = f"preempt_swap_{tier}_no_target"
            return INVALID_ACTION_PENALTY
        target = self._pick(candidates, by="size")
        self.ledger.remove_from_gpu(target)
        target.location = "queue"  # back to queue after CPU restore

        if target.current_blocks > self.ledger.cpu_free():
            self._silent_janitor(needed=target.current_blocks)

        self.ledger.place_on_cpu(target)
        # Re-add to front of appropriate queue so it gets re-admitted
        if tier == "vip":
            self.vip_queue.insert(0, target)
        else:
            self.free_queue.insert(0, target)

        self.total_swaps += 1
        penalty = PREEMPT_SWAP_FREE if tier == "free" else PREEMPT_SWAP_VIP
        self._last_action_result = f"preempted_swap_{tier}_{target.request_id}_to_cpu"
        return penalty

    def _garbage_collect(self) -> float:
        """Delete all Free-Tier CPU caches idle > 200 ticks."""
        to_remove = [
            r for r in self.ledger.cpu_requests.values()
            if r.tier == "free" and r.idle_ticks > GC_IDLE_THRESHOLD
        ]
        if not to_remove:
            self._last_action_result = "gc_removed_0_caches"
            return INVALID_ACTION_PENALTY
            
        for req in to_remove:
            self.ledger.remove_from_cpu(req)
            req.location = "evicted"
        self._last_action_result = f"gc_removed_{len(to_remove)}_caches"
        return GC_BONUS

    def _silent_janitor(self, needed: int):
        """
        CPU overflow protection. Shred oldest Free-Tier CPU caches
        until `needed` blocks are freed. Never crashes.
        """
        free_cpu = [r for r in self.ledger.cpu_requests.values() if r.tier == "free"]
        free_cpu.sort(key=lambda r: r.idle_ticks, reverse=True)  # oldest first
        freed = 0
        for req in free_cpu:
            if freed >= needed:
                break
            self.ledger.remove_from_cpu(req)
            req.location = "evicted"
            freed += req.blocks_allocated

    # ──────────────────────────────────────────────────────────────────────
    # TICK MECHANICS
    # ──────────────────────────────────────────────────────────────────────

    def _tick_forward(self) -> float:
        """
        Advance simulation by 1 tick.
        Each active request generates 1 token.
        Completed requests are harvested and rewarded.
        Idle requests accumulate idle_ticks.
        Queued requests accumulate wait_ticks.
        Returns sum of completion rewards this tick.
        """
        reward = 0.0

        # Increment wait ticks for queued requests
        for req in self.free_queue + self.vip_queue:
            req.wait_ticks += 1

        # Increment idle ticks for GPU idle and CPU requests
        for req in list(self.ledger.gpu_requests.values()):
            if req.is_idle_gpu:
                req.idle_ticks += 1

        for req in list(self.ledger.cpu_requests.values()):
            req.idle_ticks += 1

        # Generate 15 tokens per active request to speed up simulation for the demo (50 max steps)
        completed_this_tick = []
        for req in list(self.ledger.gpu_requests.values()):
            if req.is_active:
                req.generated_tokens += 5
                reward += ACTIVE_GEN_BONUS
                # Grow blocks if needed
                new_blocks = req.current_blocks
                if new_blocks > req.blocks_allocated:
                    growth = new_blocks - req.blocks_allocated
                    if growth <= self.ledger.gpu_free():
                        self.ledger.gpu_used += growth
                        req.blocks_allocated = new_blocks
                    else:
                        # Cannot grow — this causes a crash
                        self.crashed = True
                        return CRASH_PENALTY

                if req.is_complete:
                    completed_this_tick.append(req)

        # Harvest completed requests
        for req in completed_this_tick:
            total_ticks = req.wait_ticks + req.target_gen_tokens
            multiplier = VIP_MULTIPLIER if req.tier == "vip" else FREE_MULTIPLIER
            completion_reward = multiplier * math.exp(-LATENCY_DECAY * total_ticks)
            reward += completion_reward

            # Record fluency for v2 score
            fluency = req.target_gen_tokens / max(1, total_ticks)
            self._per_request_fluency.append(fluency)

            self.ledger.remove_from_gpu(req)
            req.location = "done"
            self.total_completed += 1

            # Add to returning pool so future requests can be returning users
            self._returning_pool.append(req.request_id)

        return reward

    def _spawn_traffic(self):
        """Generate new requests according to traffic function."""
        if self.tick % 5 != 0:
            return
            
        traffic_fn = TRAFFIC_FNS[self.config["traffic_fn"]]
        n_arrivals = traffic_fn(self.tick) * 5

        for _ in range(n_arrivals):
            req = generate_request(
                tick=self.tick,
                vip_ratio=self.config["vip_ratio"],
                power_user_pct=self.config["power_user_pct"],
                returning_pool=self._returning_pool,
                rng=self.rng,
            )
            self.total_arrived += 1

            if req.tier == "vip":
                if len(self.vip_queue) < VIP_QUEUE_MAX:
                    self.vip_queue.append(req)
                else:
                    # Queue full — deadlock trigger checked in _check_terminators
                    pass
            else:
                if len(self.free_queue) < FREE_QUEUE_MAX:
                    self.free_queue.append(req)

    def _apply_pressure_penalty(self) -> float:
        """
        Apply memory pressure penalty ONLY if:
        - GPU > 95% AND
        - There exist idle caches the agent could legally evict.
        """
        gpu_util = self.ledger.gpu_utilization()
        if gpu_util <= 0.95:
            return 0.0

        has_idle_targets = bool(
            self.ledger.idle_gpu_requests("free") or
            self.ledger.idle_gpu_requests("vip")
        )
        if not has_idle_targets:
            return 0.0

        if gpu_util > 0.98:
            return PAIN_98_PENALTY
        return PAIN_95_PENALTY

    def _check_sla_violations(self) -> float:
        """
        Check all queued requests for SLA timeout.
        If SLA is None (easy phase), skip.
        Returns total penalty (negative float).
        """
        sla_free = self.config["sla_free"]
        sla_vip  = self.config["sla_vip"]
        penalty  = 0.0

        if sla_free is not None:
            for req in self.free_queue:
                if req.wait_ticks >= sla_free:
                    penalty += SLA_MISS_FREE

        if sla_vip is not None:
            for req in self.vip_queue:
                if req.wait_ticks >= sla_vip:
                    penalty += SLA_MISS_VIP

        return penalty

    def _check_terminators(self) -> tuple[float, bool]:
        """
        Check for episode-ending conditions.
        Returns (penalty, should_terminate).
        """
        # Crash
        if self.crashed:
            return CRASH_PENALTY, True

        # GPU physically exceeded
        if self.ledger.gpu_used > GPU_TOTAL_BLOCKS:
            self.crashed = True
            return CRASH_PENALTY, True

        # Deadlock — both queues at capacity
        if (len(self.free_queue) >= FREE_QUEUE_MAX and
                len(self.vip_queue) >= VIP_QUEUE_MAX):
            return DEADLOCK_PENALTY, True

        return 0.0, False

    # ──────────────────────────────────────────────────────────────────────
    # OBSERVATION BUILDER
    # ──────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> "KVCacheObservation":
        """Construct the normalized 20-dimensional observation."""
        from models import KVCacheObservation

        sla_free = self.config.get("sla_free") or 200  # fallback for easy
        sla_vip  = self.config.get("sla_vip")  or 100

        # Memory pressure trend: slope of GPU utilization over last 5 ticks
        if len(self._gpu_history) >= 2:
            trend = self._gpu_history[-1] - self._gpu_history[0]
            trend = max(-1.0, min(1.0, trend))
        else:
            trend = 0.0

        # Queue pressures
        free_q_pressure = len(self.free_queue) / FREE_QUEUE_MAX
        vip_q_pressure  = len(self.vip_queue)  / VIP_QUEUE_MAX

        # Wait time percentages
        oldest_free_wait = max((r.wait_ticks for r in self.free_queue), default=0)
        oldest_vip_wait  = max((r.wait_ticks for r in self.vip_queue),  default=0)
        free_wait_pct = min(1.0, oldest_free_wait / sla_free)
        vip_wait_pct  = min(1.0, oldest_vip_wait  / sla_vip)

        # Yield signals
        active_reqs = self.ledger.active_gpu_requests()
        largest_active = max((r.current_blocks for r in active_reqs), default=0)
        yield_preempt = min(1.0, largest_active / GPU_TOTAL_BLOCKS)

        # Cache size statistics (normalized by GPU capacity)
        def _stats(reqs: list) -> tuple[float, float, float]:
            if not reqs:
                return 0.0, 0.0, 0.0
            sizes = [r.current_blocks / GPU_TOTAL_BLOCKS for r in reqs]
            return (
                min(1.0, max(sizes)),
                min(1.0, float(np.mean(sizes))),
                min(1.0, float(np.std(sizes))),
            )

        def _age_stats(reqs: list) -> tuple[float, float, float]:
            if not reqs:
                return 0.0, 0.0, 0.0
            # Normalize age by 500 ticks as reasonable max
            ages = [min(1.0, r.idle_ticks / 500) for r in reqs]
            return (
                min(1.0, max(ages)),
                min(1.0, float(np.mean(ages))),
                min(1.0, float(np.std(ages))),
            )

        free_idle = self.ledger.idle_gpu_requests("free")
        vip_idle  = self.ledger.idle_gpu_requests("vip")

        fs_max, fs_mean, fs_std = _stats(free_idle)
        vs_max, vs_mean, vs_std = _stats(vip_idle)
        fa_max, fa_mean, fa_std = _age_stats(free_idle)
        va_max, va_mean, va_std = _age_stats(vip_idle)

        return KVCacheObservation(
            gpu_utilization_pct=min(1.0, self.ledger.gpu_utilization()),
            cpu_utilization_pct=min(1.0, self.ledger.cpu_utilization()),
            memory_pressure_trend=trend,
            free_queue_pressure=min(1.0, free_q_pressure),
            vip_queue_pressure=min(1.0, vip_q_pressure),
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

    # ──────────────────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────────────────

    def _compute_final_score(self) -> float:
        """
        Final episodic score, always in [0.0, 1.0].
        v1 = throughput
        v2 = compute fluency
        v3 = hardware mastery
        """
        # v1: throughput
        v1 = self.total_completed / max(1, self.total_arrived)

        # v2: compute fluency (average per-request fluency)
        if self._per_request_fluency:
            v2 = float(np.mean(self._per_request_fluency))
        else:
            v2 = 0.0
        v2 = min(1.0, max(0.0, v2))

        # v3: hardware mastery
        cache_hit_rate = (
            self.total_cache_hits / max(1, self.total_returning_arrived)
            if self.total_returning_arrived > 0 else 0.0
        )
        swap_rate = self.total_swaps / max(1, self.total_actions)
        v3 = max(0.0, cache_hit_rate - swap_rate)
        v3 = min(1.0, v3)

        score = 0.5 * v1 + 0.3 * v2 + 0.2 * v3
        return min(1.0, max(0.0, score))

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _pick(self, reqs: list[Request], by: str) -> Request:
        """Select target request by 'size' (largest) or 'age' (oldest)."""
        if by == "size":
            return max(reqs, key=lambda r: r.current_blocks)
        else:  # age
            return max(reqs, key=lambda r: r.idle_ticks)
