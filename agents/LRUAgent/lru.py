import uuid
import time
import sys
import os

# Ensure we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from server.environment import KVCacheEnvironment

# ---------------- ACTION MAP ---------------- #

from models import ACTION_MAP
from server.env_components.scoring import compute_final_score


# ---------------- ENV WRAPPER ---------------- #

class LocalEnv:
    def __init__(self):
        self.env = KVCacheEnvironment()
        print("[*] LRUAgent: Using Local Environment")

    def reset(self, task="easy"):
        try:
            obs_obj = self.env.reset(task)
            return obs_obj.to_array()
        except Exception as e:
            print(f"[!] Reset Error: {e}")
            return None

    def step(self, action):
        try:
            obs_obj, reward, done, info = self.env.step(action)
            return obs_obj.to_array(), reward, done, info
        except Exception as e:
            print(f"[!] Step Error: {e}")
            return None, 0, True, {}

    def admit_batch(self, tier: str, pct: float) -> float:
        """Batch-admit pct% of the tier queue based on GPU space."""
        return self.env.admit_batch(tier, pct)

    def gpu_free_pct(self) -> float:
        """Return fraction of GPU blocks currently free."""
        import server.env_components.constants as constants
        gpu_total = constants.GPU_TOTAL_BLOCKS
        gpu_used = self.env.ledger.gpu_used
        return max(0.0, (gpu_total - gpu_used) / gpu_total)


# ---------------- AGENT ---------------- #

class LRUAgent:
    """
    Optimized LRU Heuristic Agent.

    Decision priority:
      1. Batch-admit from queues proportional to free GPU space (< 85%)
      2. At 85-90%: swap oldest idle caches to CPU + GC stale CPU caches
      3. At 90-95%: aggressively evict oldest idle GPU caches (LRU)
      4. At 95%+:   preempt & swap largest active requests to CPU
      5. Do-nothing only when GPU < 70% and queues are empty
    """

    def __init__(self):
        self.last_action = 17
        self.consecutive_preempt_swaps = 0
        self.consecutive_do_nothing = 0
        self.policy_mode = "normal"

    def _choose(self, action: int, mode: str = "normal") -> int:
        self.policy_mode = mode
        if action == 14:
            self.consecutive_preempt_swaps += 1
        else:
            self.consecutive_preempt_swaps = 0

        if action == 17:
            self.consecutive_do_nothing += 1
        else:
            self.consecutive_do_nothing = 0

        self.last_action = action
        return action

    def select_action(self, obs, env: "LocalEnv" = None):
        gpu_util  = obs[0]   # GPU utilization [0-1]
        cpu_util  = obs[1]   # CPU utilization [0-1]
        trend     = obs[2]   # memory pressure trend
        free_q    = obs[3]   # total_free_req  (raw count)
        vip_q     = obs[4]   # total_vip_req   (raw count)
        total_q   = obs[5]   # total queued requests
        free_wait = obs[6]   # free_max_wait_time_pct
        vip_wait  = obs[7]   # vip_max_wait_time_pct
        yield_pre = obs[8]   # largest active request size / gpu capacity
        free_age  = obs[14]  # free_age_max (oldest idle free on GPU)
        vip_age   = obs[17]  # vip_age_max  (oldest idle vip on GPU)

        has_queue = (vip_q > 0 or free_q > 0)
        has_idle_free = free_age > 0
        has_idle_vip = vip_age > 0
        has_preempt_target = yield_pre > 0.001
        heavy_backlog = total_q >= 60
        can_admit_now = (
            gpu_util < 0.74 and
            trend < 0.05 and
            not (heavy_backlog and gpu_util > 0.65)
        )

        # ── TIER 0: Batch admit if GPU has headroom ────────────────────────
        if env is not None and has_queue and can_admit_now:
            # Throttle queue admission so active-request growth has GPU headroom.
            if gpu_util < 0.55:
                cap = 0.25
            elif gpu_util < 0.68:
                cap = 0.12
            else:
                cap = 0.05

            admit_pct = min(cap, env.gpu_free_pct())
            if admit_pct > 0.0:
                if vip_q > 0:
                    env.admit_batch("vip", admit_pct)
                if free_q > 0:
                    env.admit_batch("free", admit_pct)

        # Recovery lane: if we repeatedly preempted and pressure dropped a bit,
        # force one admission to avoid preempt-only loops that starve completion.
        if self.consecutive_preempt_swaps >= 3 and gpu_util < 0.88 and has_queue:
            if can_admit_now:
                if vip_q > 0 and vip_wait >= 0.25:
                    return self._choose(9, "recovery")
                if vip_q > 0:
                    return self._choose(9, "recovery")
                if free_q > 0:
                    return self._choose(8, "recovery")
            if has_idle_free and cpu_util < 0.80:
                return self._choose(6, "recovery")
            if has_idle_vip and cpu_util < 0.80:
                return self._choose(7, "recovery")
            if has_preempt_target:
                return self._choose(14, "recovery")
            return self._choose(16, "recovery")

        # ── TIER 1: Critical — preempt active requests to CPU (95%+) ───────
        if gpu_util > 0.95:
            if has_preempt_target:
                return self._choose(14, "safety")  # Preempt & Swap Largest Active Free -> CPU
            if has_idle_free and cpu_util < 0.85:
                return self._choose(6, "safety")   # Swap Oldest Free cache GPU->CPU
            if has_idle_vip and cpu_util < 0.85:
                return self._choose(7, "safety")   # Swap Oldest VIP cache GPU->CPU
            if has_idle_free:
                return self._choose(2, "safety")   # Evict Oldest Free cache
            if has_idle_vip:
                return self._choose(3, "safety")   # Evict Oldest VIP cache
            return self._choose(16, "safety")      # Last resort: GC

        # Trend-aware early intervention before hard saturation.
        if trend > 0.15 and gpu_util > 0.82:
            if has_idle_free and cpu_util < 0.85:
                return self._choose(6, "trend")
            if has_preempt_target:
                return self._choose(14, "trend")

        if trend > 0.10 and gpu_util > 0.78:
            if has_idle_free and cpu_util < 0.80:
                return self._choose(6, "trend")
            if has_idle_vip and cpu_util < 0.80:
                return self._choose(7, "trend")

        # ── TIER 2: High pressure — evict + swap idle caches (90-95%) ──────
        if gpu_util > 0.90:
            if has_idle_free and cpu_util < 0.80:
                return self._choose(6, "pressure")   # Swap Oldest Free cache GPU->CPU
            if has_idle_vip and cpu_util < 0.80:
                return self._choose(7, "pressure")   # Swap Oldest VIP cache GPU->CPU
            if has_idle_vip and vip_age >= free_age:
                return self._choose(3, "pressure")   # Evict Oldest VIP (idle GPU)
            if has_idle_free:
                return self._choose(2, "pressure")   # Evict Oldest Free (idle GPU)
            if has_preempt_target:
                return self._choose(14, "pressure")  # Preempt & Swap Largest Active Free -> CPU
            return self._choose(16, "pressure")      # GC

        # ── TIER 3: Medium pressure — offload to CPU (85-90%) ──────────────
        if gpu_util > 0.85:
            if has_idle_free and cpu_util < 0.80:
                return self._choose(6, "pressure")
            if has_idle_vip and cpu_util < 0.80:
                return self._choose(7, "pressure")
            if cpu_util >= 0.80:
                return self._choose(16, "pressure")
            if has_preempt_target:
                return self._choose(14, "pressure")
            return self._choose(16, "pressure")

        # ── TIER 4: Low-medium (70-85%) — proactive CPU GC + SLA triage ───
        if gpu_util > 0.70:
            if cpu_util > 0.60:
                return self._choose(16, "normal")
            if free_age > 0 and cpu_util < 0.75:
                return self._choose(6, "normal")
            if vip_wait > 0.5 and vip_q > 0:
                return self._choose(9, "normal")

            # Safety gate: avoid idle action with pending queue near high util.
            if gpu_util >= 0.82 and has_queue:
                if has_idle_free and cpu_util < 0.80:
                    return self._choose(6, "safety")
                if has_preempt_target:
                    return self._choose(14, "safety")
                return self._choose(16, "safety")

            if has_queue and can_admit_now:
                return self._choose(9 if vip_q > 0 else 8, "normal")
            if has_queue and has_preempt_target and gpu_util > 0.76:
                return self._choose(14, "normal")
            return self._choose(17, "normal")

        # ── TIER 5: GPU comfortably free (< 70%) ───────────────────────────
        if cpu_util > 0.50:
            return self._choose(16, "normal")
        if vip_q > 0 and can_admit_now:
            return self._choose(9, "normal")
        if free_q > 0 and can_admit_now:
            return self._choose(8, "normal")
        if has_queue and has_preempt_target and gpu_util > 0.72:
            return self._choose(14, "normal")
        return self._choose(17, "normal")



# ---------------- RUN ---------------- #

def run_sim(task=None, ticks=None):
    env = LocalEnv()
    agent = LRUAgent()
    
    all_tick_logs = []
    all_session_logs = []
    keys = [
        "gpu_utilization_pct",
        "cpu_utilization_pct",
        "memory_pressure_trend",
        "total_free_req",
        "total_vip_req",
        "total_req",
        "free_max_wait_time_pct",
        "vip_max_wait_time_pct",
        "yield_preempt_active",
        "free_size_max",
        "free_size_mean",
        "free_size_std_dev",
        "vip_size_max",
        "vip_size_mean",
        "vip_size_std_dev",
        "free_age_max",
        "free_age_mean",
        "free_age_std_dev",
        "vip_age_max",
        "vip_age_mean",
        "vip_age_std_dev",
    ]
    if task is None:
        tasks_to_run = ["easy", "medium", "hard"]
        display_task = "MIXED"
    else:
        tasks_to_run = [task]
        display_task = task.upper()
    for ep, current_task in enumerate(tasks_to_run, start=1):
        sessionID = str(uuid.uuid4())
        obs = env.reset(current_task)
        if ticks is not None:
            env.env.config["max_ticks"] = ticks
        
        if obs is None:
            print(f"[!] Reset failed for task '{current_task}', skipping.")
            continue

        total_reward = 0
        ticks_run = 0
        done = False
        logs = []
        while not done:
            action = agent.select_action(obs, env=env)
            action_name = ACTION_MAP.get(action, "Unknown")

            obs, reward, done, info = env.step(action)
            if obs is None:
                done = True
                break

            total_reward += reward
            ticks_run += 1
            
            
            obs_dict = dict(zip(keys, obs))
            log_entry = {
                "task": current_task,
                "tick": ticks_run,
                "session_id": sessionID,
                "action": action_name,
                "policy_mode": agent.policy_mode,
                "preempt_swap_streak": agent.consecutive_preempt_swaps,
                "reward": round(reward, 2),
                "score": round(total_reward, 2),
                "tick_prompt_tokens": info.get("tick_prompt_tokens", 0),
                "tick_gen_tokens": info.get("tick_gen_tokens", 0),
                "tick_max_tokens": info.get("tick_max_tokens", 0),
                **obs_dict
            }

            logs.append(log_entry)
            # print(log_entry)

        final_score = compute_final_score(
            task=current_task,
            total_completed=env.env.total_completed,
            total_arrived=env.env.total_arrived,
            per_request_fluency=env.env._per_request_fluency,
            total_cache_hits=env.env.total_cache_hits,
            total_returning_arrived=env.env.total_returning_arrived,
            total_swaps=env.env.total_swaps,
            total_actions=env.env.total_actions,
        )

        
        session_log = {
            "session_id": sessionID,
            "task": current_task,
            "total_reward": total_reward,
            "final_score": final_score,
            "ticks_run": ticks_run,
            "total_arrived": env.env.total_arrived,
            "total_completed": env.env.total_completed,
            "crashed": getattr(env.env, 'crashed', False)
        }
        all_session_logs.append(session_log)
        all_tick_logs.extend(logs)
        
        time.sleep(1)

    return all_tick_logs, all_session_logs


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else None
    run_sim(task=task)