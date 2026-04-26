import time
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import ACTION_MAP
from server.environment import KVCacheEnvironment


# ---------------- ENV ---------------- #

class LocalEnv:
    def __init__(self):
        self.env = KVCacheEnvironment()
        print("[*] Optimized Learning Agent (Stable RL)")

    def reset(self, task="easy"):
        obs = self.env.reset(task)
        return obs.to_array()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.to_array(), reward, done, info


# ---------------- AGENT ---------------- #

class OptimizedQLearningAgent:
    def __init__(self, action_size=18):
        self.action_size = action_size
        self.q_table = {}

        # Better hyperparameters
        self.lr = 0.05
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05

        # Experience replay
        self.memory = []
        self.batch_size = 64

    def discretize(self, obs):
        """
        Coarse state space representation to enable Tabular Q-Learning to converge
        within a few episodes.
        """
        return (
            # 1. GPU Utilization: 0=Low (<0.8), 1=High (0.8-0.95), 2=Critical (>0.95)
            0 if obs[0] < 0.8 else (1 if obs[0] < 0.95 else 2),
            
            # 2. VIP Queue: 0=Empty, 1=Has Items
            1 if obs[4] > 0 else 0,
            
            # 3. Free Queue: 0=Empty, 1=Has Items
            1 if obs[3] > 0 else 0,
            
            # 4. VIP Idle Caches Available? (VIP Age Max > 0)
            1 if obs[17] > 0 else 0,
            
            # 5. Free Idle Caches Available? (Free Age Max > 0)
            1 if obs[14] > 0 else 0,
        )

    def _init_state(self, state):
        """
        Pessimistic initialization to prevent the agent from picking suicidal
        actions just because their Q-value is 0. We gently bias it towards
        admitting and doing nothing first.
        """
        if state not in self.q_table:
            # Initialize all to very negative so untried actions aren't favored
            q = np.full(self.action_size, -10000.0)
            # Give a slight optimistic boost to normal operations (Admit, Do Nothing)
            q[8] = -5000.0  # Admit Free
            q[9] = -5000.0  # Admit VIP
            q[17] = -5000.0 # Do Nothing
            self.q_table[state] = q

    def select_action(self, obs):
        state = self.discretize(obs)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        self._init_state(state)

        return int(np.argmax(self.q_table[state]))

    def store(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for obs, action, reward, next_obs, done in batch:
            state = self.discretize(obs)
            next_state = self.discretize(next_obs)

            self._init_state(state)
            self._init_state(next_state)

            # REMOVED CLIPPING: Let the agent feel the full -1000 crash penalties!
            
            q_predict = self.q_table[state][action]

            if done:
                q_target = reward
            else:
                q_target = reward + self.gamma * np.max(self.q_table[next_state])

            self.q_table[state][action] += self.lr * (q_target - q_predict)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


from server.env_components.scoring import compute_final_score
import uuid

# ---------------- RUN ---------------- #

def run_sim(task=None, ticks=None):
    env = LocalEnv()
    agent = OptimizedQLearningAgent()

    print("\n" + "="*60)
    print(" OPTIMIZED LEARNING AGENT TRAINING")
    print("="*60 + "\n")

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
    else:
        tasks_to_run = [task]

    for current_task in tasks_to_run:
        sessionID = str(uuid.uuid4())
        obs = env.reset(current_task)
        if ticks is not None:
            env.env.config["max_ticks"] = ticks
            
        if obs is None:
            print(f"[!] Reset failed for task '{current_task}', skipping.")
            continue

        total_reward = 0
        done = False
        ticks_run = 0
        logs = []

        while not done:
            action = agent.select_action(obs)
            action_name = ACTION_MAP.get(action, "Unknown")

            next_obs, reward, done, info = env.step(action)

            agent.store((obs, action, reward, next_obs, done))
            agent.train()

            obs_dict = dict(zip(keys, obs))
            log_entry = {
                "task": current_task,
                "tick": ticks_run,
                "session_id": sessionID,
                "action": action_name,
                "reward": round(reward, 2),
                "score": round(total_reward + reward, 2),
                "episode": 1,
                "tick_prompt_tokens": info.get("tick_prompt_tokens", 0),
                "tick_gen_tokens": info.get("tick_gen_tokens", 0),
                "tick_max_tokens": info.get("tick_max_tokens", 0),
                **obs_dict
            }
            logs.append(log_entry)

            obs = next_obs
            total_reward += reward
            ticks_run += 1

            if ticks_run % 20 == 0 or done:
                print(
                    f"[{current_task.upper()} EP 1] Tick {ticks_run:3} | "
                    f"{action_name:25} | "
                    f"Reward {reward:+.2f} | "
                    f"Total {total_reward:.2f} | "
                    f"Eps {agent.epsilon:.3f}"
                )

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
            "episode": 1,
            "total_reward": total_reward,
            "final_score": final_score,
            "ticks_run": ticks_run,
            "total_arrived": env.env.total_arrived,
            "total_completed": env.env.total_completed,
            "crashed": getattr(env.env, 'crashed', False)
        }
        
        all_session_logs.append(session_log)
        all_tick_logs.extend(logs)

        print(f"\n[✓] {current_task.upper()} Episode 1 Score: {total_reward:.2f} | Final Env Score: {final_score:.3f}")
        print("-"*50)

    return all_tick_logs, all_session_logs

if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_sim(task=task_arg)