import os
import sys
import uuid
import time
import random
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import ACTION_MAP
from server.environment import KVCacheEnvironment
from server.env_components.scoring import compute_final_score

# ---------------- LOCAL ENV WRAPPER ---------------- #
class LocalEnv:
    def __init__(self):
        self.env = KVCacheEnvironment()
        print("[*] NeuralAgent (DQN): Using Local Environment")

    def reset(self, task="easy"):
        obs_obj = self.env.reset(task)
        if obs_obj is None: return None
        return obs_obj.to_array()

    def step(self, action):
        obs_obj, reward, done, info = self.env.step(action)
        if obs_obj is None: return None, reward, done, info
        return obs_obj.to_array(), reward, done, info

# ---------------- NEURAL NETWORK ---------------- #
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- REPLAY BUFFER ---------------- #
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ---------------- AGENT ---------------- #
class DQNAgent:
    def __init__(self, state_dim=21, action_dim=18):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] DQN using device: {self.device}")

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s', a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss and optimization
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

        self.steps_done += 1
        
        # Decay Epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------------- API RUNNER ---------------- #
def run_sim(task=None, ticks=None):
    env = LocalEnv()
    agent = DQNAgent()

    print("\n" + "="*60)
    print(" NEURAL NETWORK AGENT (DQN) TRAINING")
    print("="*60 + "\n")

    all_tick_logs = []
    all_session_logs = []
    
    keys = [
        "gpu_utilization_pct", "cpu_utilization_pct", "memory_pressure_trend",
        "total_free_req", "total_vip_req", "total_req", "free_max_wait_time_pct",
        "vip_max_wait_time_pct", "yield_preempt_active", "free_size_max",
        "free_size_mean", "free_size_std_dev", "vip_size_max", "vip_size_mean",
        "vip_size_std_dev", "free_age_max", "free_age_mean", "free_age_std_dev",
        "vip_age_max", "vip_age_mean", "vip_age_std_dev"
    ]

    tasks_to_run = ["easy", "medium", "hard"] if task is None else [task]

    for current_task in tasks_to_run:
        sessionID = str(uuid.uuid4())
        obs = env.reset(current_task)
        if ticks is not None:
            env.env.config["max_ticks"] = ticks
            
        if obs is None: break

        total_reward = 0
        ticks_run = 0
        done = False
        logs = []

        while not done:
            action = agent.select_action(obs)
            action_name = ACTION_MAP.get(action, "Unknown")

            next_obs, reward, done, info = env.step(action)
            if next_obs is None: break

            agent.memory.push(obs, action, reward, next_obs, done)
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
                print(f"[{current_task.upper()} EP 1] Tick {ticks_run:3} | {action_name:25} | Reward {reward:+.2f} | Total {total_reward:.2f} | Eps {agent.epsilon:.3f}")

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
            "session_id": sessionID, "task": current_task, "episode": 1,
            "total_reward": total_reward, "final_score": final_score,
            "ticks_run": ticks_run, "total_arrived": env.env.total_arrived,
            "total_completed": env.env.total_completed, 
            "crashed": getattr(env.env, 'crashed', False)
        }
        all_session_logs.append(session_log)
        all_tick_logs.extend(logs)

        print(f"\n[✓] {current_task.upper()} Episode 1 Score: {total_reward:.2f} | Final Env Score: {final_score:.3f}\n" + "-"*50)

    return all_tick_logs, all_session_logs

if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_sim(task=task_arg)
