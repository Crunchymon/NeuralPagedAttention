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
    Least Recently Used (Age-based) Heuristic Agent.
    Admission is batch-based: admits GPU-free-pct% of both queues each step.
    """

    def select_action(self, obs, env: "LocalEnv" = None):
        gpu_util = obs[0]
        free_q   = obs[3]   # total_free_req (raw count)
        vip_q    = obs[4]   # total_vip_req
        free_age = obs[14]
        vip_age  = obs[17]

        # --- Batch admit proportional to GPU free space ---
        if env is not None and (free_q > 0 or vip_q > 0) and gpu_util < 0.85:
            admit_pct = env.gpu_free_pct()          # e.g. 0.40 → admit 40% of queue
            if vip_q > 0:
                env.admit_batch("vip", admit_pct)
            if free_q > 0:
                env.admit_batch("free", admit_pct)
            # Fall through to normal housekeeping action

        # --- LRU eviction under pressure ---
        if gpu_util > 0.90:
            if vip_age >= free_age and vip_age > 0:
                return 3   # evict_oldest_vip
            elif free_age > 0:
                return 2   # evict_oldest_free
            if gpu_util > 0.95:
                return 14  # preempt_swap_largest_free -> CPU
            return 16      # garbage_collect

        return 17  # do_nothing


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
            break

        total_reward = 0
        ticks_run = 0
        done = False
        logs = []
        while not done:
            action = agent.select_action(obs, env=env)
            action_name = ACTION_MAP.get(action, "Unknown")

            obs, reward, done, info = env.step(action)
            if obs is None:
                break

            total_reward += reward
            ticks_run += 1
            
            
            obs_dict = dict(zip(keys, obs))
            log_entry = {
                "task": current_task,
                "tick": ticks_run,
                "session_id": sessionID,
                "action": action_name,
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