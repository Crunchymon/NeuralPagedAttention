import random
import uuid

from models import KVCacheObservation, KVCacheState
from server.env_components.action_executor import execute_action, admit_batch as _admit_batch_fn
from server.env_components.constants import (
    CRASH_PENALTY,
    FREE_QUEUE_MAX,
    PHASE_CONFIGS,
    VIP_QUEUE_MAX,
)
import server.env_components.constants as constants
from server.env_components.observation import build_observation
from server.env_components.penalties import (
    apply_pressure_penalty,
    check_sla_violations,
    check_terminators,
)
from server.env_components.request_factory import generate_request
from server.env_components.scoring import compute_final_score
from server.env_components.state import MemoryLedger, Request
from server.env_components.tick_engine import tick_forward
from server.env_components.traffic import TRAFFIC_FNS


class KVCacheEnvironment:
    """Orchestrates environment components and episode lifecycle."""

    def __init__(self):
        self.task: str = "easy"
        self.config: dict = PHASE_CONFIGS[self.task]
        self.tick: int = 0
        self.episode_id: str = ""
        self.rng: random.Random = random.Random()

        self.free_queue: list[Request] = []
        self.vip_queue: list[Request] = []
        self.ledger: MemoryLedger = MemoryLedger()

        self.total_arrived: int = 0
        self.total_completed: int = 0
        self.total_rejected: int = 0
        self.total_swaps: int = 0
        self.total_actions: int = 0
        self.total_cache_hits: int = 0
        self.total_returning_arrived: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.crashed: bool = False

        self._per_request_fluency: list[float] = []
        self._gpu_history: list[float] = []
        self._returning_pool: list[str] = []
        self._last_action_result: str = "none"

        # Per-tick token stats (reset each _spawn_traffic call)
        self.tick_prompt_tokens: int = 0
        self.tick_gen_tokens: int = 0
        self.tick_max_tokens: int = 0

    def reset(self, task: str = "easy") -> "KVCacheObservation":
        assert task in PHASE_CONFIGS, f"Unknown task: {task}"
        self.task = task
        self.config = PHASE_CONFIGS[task]
        self.tick = 0
        self.episode_id = str(uuid.uuid4())[:12]
        self.rng = random.Random(42)

        self.free_queue = []
        self.vip_queue = []
        self.ledger = MemoryLedger()

        self.total_arrived = 0
        self.total_completed = 0
        self.total_rejected = 0
        self.total_swaps = 0
        self.total_actions = 0
        self.total_cache_hits = 0
        self.total_returning_arrived = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.crashed = False

        self._per_request_fluency = []
        self._gpu_history = []
        self._returning_pool = []
        self._last_action_result = "reset"

        self.tick_prompt_tokens = 0
        self.tick_gen_tokens = 0
        self.tick_max_tokens = 0

        self._spawn_traffic()
        return self._build_observation()

    def step(self, action_id: int) -> tuple:
        if self.done:
            return self._build_observation(), 0.0, True, {"error": "episode_done"}

        reward = 0.0
        self.total_actions += 1

        reward += self._execute_action(action_id)
        reward += self._tick_forward()
        self._spawn_traffic()
        reward += self._apply_pressure_penalty()

        sla_penalty = self._check_sla_violations()
        reward += sla_penalty

        terminator_penalty, terminated = self._check_terminators()
        reward += terminator_penalty

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
            "tick_prompt_tokens": self.tick_prompt_tokens,
            "tick_gen_tokens": self.tick_gen_tokens,
            "tick_max_tokens": self.tick_max_tokens,
        }

        return self._build_observation(), reward, self.done, info

    def state(self) -> "KVCacheState":
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

    def _execute_action(self, action_id: int) -> float:
        return execute_action(self, action_id)

    def _tick_forward(self) -> float:
        return tick_forward(self)

    def admit_batch(self, tier: str, pct: float) -> float:
        """Admit up to pct% of the tier queue subject to GPU availability."""
        return _admit_batch_fn(self, tier, pct)

    def _spawn_traffic(self):
        if self.tick % 5 != 0:
            return

        # Reset per-tick token counters for this batch
        self.tick_prompt_tokens = 0
        self.tick_gen_tokens = 0
        self.tick_max_tokens = 0

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

            # Accumulate token stats
            max_tokens = int(constants.GPU_TOTAL_BLOCKS * 0.8 * constants.TOKENS_PER_BLOCK)
            self.tick_prompt_tokens += req.prompt_tokens
            self.tick_gen_tokens += req.target_gen_tokens
            self.tick_max_tokens += max_tokens

            if req.tier == "vip":
                if len(self.vip_queue) < VIP_QUEUE_MAX:
                    self.vip_queue.append(req)
            else:
                if len(self.free_queue) < FREE_QUEUE_MAX:
                    self.free_queue.append(req)

    def _apply_pressure_penalty(self) -> float:
        return apply_pressure_penalty(
            gpu_util=self.ledger.gpu_utilization(),
            has_idle_targets=bool(
                self.ledger.idle_gpu_requests("free") or
                self.ledger.idle_gpu_requests("vip")
            ),
        )

    def _check_sla_violations(self) -> float:
        return check_sla_violations(
            free_queue=self.free_queue,
            vip_queue=self.vip_queue,
            sla_free=self.config["sla_free"],
            sla_vip=self.config["sla_vip"],
        )

    def _check_terminators(self) -> tuple[float, bool]:
        penalty, terminate = check_terminators(
            crashed=self.crashed,
            gpu_used=self.ledger.gpu_used,
            gpu_total=constants.GPU_TOTAL_BLOCKS,
            free_queue_len=len(self.free_queue),
            vip_queue_len=len(self.vip_queue),
        )
        if terminate:
            self.crashed = True  # crashed on any early termination (GPU overflow or deadlock)
        return penalty, terminate

    def _build_observation(self) -> "KVCacheObservation":
        return build_observation(
            free_queue=self.free_queue,
            vip_queue=self.vip_queue,
            ledger=self.ledger,
            gpu_history=self._gpu_history,
            gpu_total_blocks=constants.GPU_TOTAL_BLOCKS,
            free_queue_max=FREE_QUEUE_MAX,
            vip_queue_max=VIP_QUEUE_MAX,
            sla_free=self.config.get("sla_free") or 200,
            sla_vip=self.config.get("sla_vip") or 100,
        )

    def _compute_final_score(self) -> float:
        return compute_final_score(
            task=self.task,
            total_completed=self.total_completed,
            total_arrived=self.total_arrived,
            per_request_fluency=self._per_request_fluency,
            total_cache_hits=self.total_cache_hits,
            total_returning_arrived=self.total_returning_arrived,
            total_swaps=self.total_swaps,
            total_actions=self.total_actions,
        )
