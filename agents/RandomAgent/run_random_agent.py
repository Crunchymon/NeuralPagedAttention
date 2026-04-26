import random
import time
import sys
import os
import uuid

# Ensure we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import ACTION_MAP
from server.environment import KVCacheEnvironment
from server.env_components.scoring import compute_final_score


# ---------------- ENV WRAPPER ---------------- #

class RemoteEnv:
    def __init__(self):
        self.env = KVCacheEnvironment()
        print(f"[*] Initialized Local Environment")

    def reset(self, task="easy", seed=None, traffic_trace=None):
        try:
            obs_obj = self.env.reset(task, seed=seed, traffic_trace=traffic_trace)

            # Convert observation object → list
            obs = obs_obj.to_array() if hasattr(obs_obj, "to_array") else obs_obj

            print(obs)
            return obs

        except Exception as e:
            print(f"[!] Error resetting environment: {e}")
            return None

    def step(self, action):
        try:
            obs_obj, reward, done, info = self.env.step(action)

            obs = obs_obj.to_array() if hasattr(obs_obj, "to_array") else obs_obj

            return obs, reward, done, info

        except Exception as e:
            print(f"[!] Error stepping environment: {e}")
            return None, 0, True, {"error": str(e)}

    def close(self):
        # Nothing needed for local env
        pass

    def admit_batch(self, tier: str, pct: float) -> float:
        """Batch-admit pct% of the tier queue based on GPU space."""
        return self.env.admit_batch(tier, pct)


# ---------------- AGENT ---------------- #

class RandomAgent:
    def __init__(self, action_size=18):
        self.action_size = action_size

    def select_action(self, state, env: "RemoteEnv" = None):
        """Pick a random action. Before that, randomly admit a pct of both queues."""
        if env is not None:
            pct = random.uniform(0.1, 1.0)  # random 10%-100% of each queue
            free_q = state[3]               # total_free_req
            vip_q  = state[4]               # total_vip_req
            if vip_q > 0:
                env.admit_batch("vip", pct)
            if free_q > 0:
                env.admit_batch("free", pct)
        return random.randint(0, self.action_size - 1)


# ---------------- RUN ---------------- #

def run_simulation(task=None, ticks=None, seed=None, traffic_trace=None):
    env = RemoteEnv()
    agent = RandomAgent()

    all_tick_logs = []
    all_session_logs = []
    
    keys = [
        "gpu_utilization_pct", "cpu_utilization_pct", "memory_pressure_trend",
        "total_free_req", "total_vip_req", "total_req", "free_max_wait_time_pct",
        "vip_max_wait_time_pct", "yield_preempt_active", "free_size_max",
        "free_size_mean", "free_size_std_dev", "vip_size_max", "vip_size_mean",
        "vip_size_std_dev", "free_age_max", "free_age_mean", "free_age_std_dev",
        "vip_age_max", "vip_age_mean", "vip_age_std_dev",
    ]

    if task is None:
        tasks_to_run = ["easy", "medium", "hard"]
        display_task = "MIXED"
    else:
        tasks_to_run = [task]
        display_task = task.upper()

    print(f"\n{'='*50}")
    print(f" RANDOM AGENT BASELINE: {display_task} TASK")
    print(f"{'='*50}\n")

    episode_rewards = []

    for ep, current_task in enumerate(tasks_to_run, start=1):
        sessionID = str(uuid.uuid4())
        print(f"[@] Starting Episode {ep} ({current_task.upper()})...")

        obs = env.reset(task=current_task, seed=seed, traffic_trace=traffic_trace)
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
            current_state_dict = dict(zip(keys, obs))
            action = agent.select_action(obs, env=env)
            action_name = ACTION_MAP.get(action, f"Unknown({action})")

            obs, reward, done, info = env.step(action)

            if obs is None:
                break

            total_reward += reward
            ticks_run += 1
            
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
                **current_state_dict
            }
            logs.append(log_entry)

        episode_rewards.append(total_reward)
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

        print(
            f"\n[✓] Episode {ep} Complete. "
            f"Ticks: {ticks} | Final Score: {final_score:.4f}\n{'-'*50}\n"
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

    env.close()
    return all_tick_logs, all_session_logs


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_simulation(task=task_arg)