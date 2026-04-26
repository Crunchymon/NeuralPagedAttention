from __future__ import annotations

import os
import sys
import uuid
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import ACTION_MAP
from server.environment import KVCacheEnvironment
from server.env_components.scoring import compute_final_score
import server.env_components.constants as constants


def normalize_obs(obs: list[float]) -> np.ndarray:
    """Scale raw queue counts to ~[0,1] for stable Q-learning."""
    arr = np.asarray(obs, dtype=np.float32).copy()
    arr[3] = min(1.0, arr[3] / float(constants.FREE_QUEUE_MAX))
    arr[4] = min(1.0, arr[4] / float(constants.VIP_QUEUE_MAX))
    arr[5] = min(1.0, arr[5] / float(constants.FREE_QUEUE_MAX + constants.VIP_QUEUE_MAX))
    return arr


def pre_step_admit(env: "LocalEnv", obs: list[float]) -> None:
    """Match LRU/LLM/PPO: admit when GPU has headroom."""
    gpu_util, free_q, vip_q = obs[0], obs[3], obs[4]
    if gpu_util < 0.85 and (vip_q > 0 or free_q > 0):
        pct = env.gpu_free_pct()
        if vip_q > 0:
            env.admit_batch("vip", pct)
        if free_q > 0:
            env.admit_batch("free", pct)


class LocalEnv:
    def __init__(self):
        self.env = KVCacheEnvironment()
        print("[*] NeuralAgent (DQN): Using Local Environment")

    def reset(self, task="easy"):
        obs_obj = self.env.reset(task)
        if obs_obj is None:
            return None
        return obs_obj.to_array()

    def step(self, action):
        obs_obj, reward, done, info = self.env.step(action)
        if obs_obj is None:
            return None, reward, done, info
        return obs_obj.to_array(), reward, done, info

    def admit_batch(self, tier: str, pct: float) -> float:
        return self.env.admit_batch(tier, pct)

    def gpu_free_pct(self) -> float:
        gpu_total = constants.GPU_TOTAL_BLOCKS
        gpu_used = self.env.ledger.gpu_used
        return max(0.0, (gpu_total - gpu_used) / gpu_total)


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer: list = []
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


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 18,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[*] DQN using device: {self.device}")

        self.policy_net = DQN(state_dim, action_dim, hidden).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = ReplayBuffer(capacity=200_000)

        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = 500_000
        self.total_env_steps = 0
        self.train_updates = 0
        self.grad_clip = 10.0

    def _epsilon_by_step(self) -> float:
        if self.total_env_steps >= self.epsilon_decay_steps:
            return self.epsilon_min
        frac = 1.0 - (self.total_env_steps / float(self.epsilon_decay_steps))
        return self.epsilon_min + (1.0 - self.epsilon_min) * frac

    def select_action(self, state_norm: np.ndarray, greedy: bool = False):
        sched = self._epsilon_by_step()
        self.epsilon = sched
        eps = 0.0 if greedy else sched
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax().item())

    def train_step(self):
        if len(self.memory) < 50_000:
            return None
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            expected_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_values, expected_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_updates += 1

        with torch.no_grad():
            for tp, sp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

        return float(loss.item())

    def save(self, filepath: str):
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_env_steps": self.total_env_steps,
                "train_updates": self.train_updates,
            },
            filepath,
        )

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            self.policy_net.load_state_dict(ckpt["policy"])
            if "target" in ckpt:
                self.target_net.load_state_dict(ckpt["target"])
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
            self.total_env_steps = ckpt.get("total_env_steps", 0)
            self.train_updates = ckpt.get("train_updates", 0)
        else:
            self.policy_net.load_state_dict(ckpt)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.05
        print(f"[*] Loaded weights from {filepath} (steps={self.total_env_steps}).")
        return True


def _eval_agent(agent: DQNAgent, tasks: list[str], max_ticks_cap: int | None = None) -> float:
    agent.policy_net.eval()
    scores = []
    for task in tasks:
        env = LocalEnv()
        obs = env.reset(task)
        if obs is None:
            continue
        if max_ticks_cap is not None:
            env.env.config["max_ticks"] = min(env.env.config["max_ticks"], max_ticks_cap)
        done = False
        while not done:
            pre_step_admit(env, obs)
            a = agent.select_action(normalize_obs(obs), greedy=True)
            nobs, _, done, _ = env.step(a)
            if nobs is None:
                break
            obs = nobs
        scores.append(
            compute_final_score(
                task=task,
                total_completed=env.env.total_completed,
                total_arrived=env.env.total_arrived,
                per_request_fluency=env.env._per_request_fluency,
                total_cache_hits=env.env.total_cache_hits,
                total_returning_arrived=env.env.total_returning_arrived,
                total_swaps=env.env.total_swaps,
                total_actions=env.env.total_actions,
            )
        )
    agent.policy_net.train()
    return float(np.mean(scores)) if scores else 0.0


def train_offline(
    episodes: int = 400,
    warmup_steps: int = 50_000,
    eval_every: int = 40,
    seed: int = 42,
    fast: bool = False,
    resume: str | None = None,
):
    if fast:
        warmup_steps = 2_000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = LocalEnv()
    agent = DQNAgent()

    weights_path = os.path.join(os.path.dirname(__file__), "dqn_weights.pth")
    best_path = os.path.join(os.path.dirname(__file__), "dqn_weights_best.pth")

    if resume:
        agent.load(resume)
    elif os.path.exists(weights_path):
        agent.load(weights_path)

    print("\n" + "=" * 60)
    print(" DQN TRAINING (Double DQN + soft targets + normalized obs)")
    print("=" * 60 + "\n")

    curriculum = ["easy"] * min(80, episodes // 4) if not fast else ["easy"] * 5
    rest = max(0, episodes - len(curriculum))
    task_pool = curriculum + (["easy", "medium", "hard"] * ((rest + 2) // 3))[:rest]

    best_eval = -1.0
    loss_ema = None

    for episode in range(episodes):
        current_task = task_pool[episode] if episode < len(task_pool) else random.choice(["easy", "medium", "hard"])
        obs = env.reset(current_task)
        if obs is None:
            continue

        if fast:
            env.env.config["max_ticks"] = min(env.env.config["max_ticks"], 400)

        done = False
        ep_reward = 0.0

        while not done:
            pre_step_admit(env, obs)
            s = normalize_obs(obs)
            action = agent.select_action(s)
            next_obs, reward, done, _ = env.step(action)
            if next_obs is None:
                break

            sn = normalize_obs(next_obs)
            r = float(np.clip(reward, -50.0, 50.0))
            agent.memory.push(s, action, r, sn, float(done))
            agent.total_env_steps += 1

            if len(agent.memory) >= warmup_steps:
                loss = agent.train_step()
                if loss is not None:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss if loss_ema is not None else loss

            obs = next_obs
            ep_reward += reward

        if (episode + 1) % max(1, eval_every) == 0:
            cap = 600 if fast else None
            ev = _eval_agent(agent, ["easy", "medium", "hard"], max_ticks_cap=cap)
            print(
                f"[eval] ep {episode + 1}/{episodes} | mean_final_score={ev:.4f} | "
                f"steps={agent.total_env_steps} | eps(train)={agent._epsilon_by_step():.3f} | loss_ema={loss_ema}"
            )
            if ev > best_eval:
                best_eval = ev
                agent.save(best_path)
                print(f"  → new best checkpoint saved ({ev:.4f})")

        if (episode + 1) % 20 == 0:
            print(
                f"[train] ep {episode + 1}/{episodes} | task={current_task} | "
                f"ep_return={ep_reward:.1f} | steps={agent.total_env_steps} | eps={agent.epsilon:.3f}"
            )

    agent.save(weights_path)
    print(f"\n[*] Training complete. Last checkpoint: {weights_path}")
    if best_eval >= 0:
        print(f"[*] Best eval mean score: {best_eval:.4f} → {best_path}")


def run_sim(task=None, ticks=None):
    env = LocalEnv()
    agent = DQNAgent()

    weights_path = os.path.join(os.path.dirname(__file__), "dqn_weights.pth")
    best_path = os.path.join(os.path.dirname(__file__), "dqn_weights_best.pth")
    loaded = False
    if os.path.exists(best_path):
        loaded = agent.load(best_path)
    if not loaded:
        loaded = agent.load(weights_path)

    print("\n" + "=" * 60)
    print(f" NEURAL NETWORK AGENT (DQN) {'(checkpoint)' if loaded else '(random init — train first)'}")
    print("=" * 60 + "\n")

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

    tasks_to_run = ["easy", "medium", "hard"] if task is None else [task]

    for current_task in tasks_to_run:
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
            pre_step_admit(env, obs)
            s = normalize_obs(obs)
            action = agent.select_action(s)
            action_name = ACTION_MAP.get(action, "Unknown")

            next_obs, reward, done, info = env.step(action)
            if next_obs is None:
                break

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
                **obs_dict,
            }
            logs.append(log_entry)

            obs = next_obs
            total_reward += reward
            ticks_run += 1

            if ticks_run % 20 == 0 or done:
                print(
                    f"[{current_task.upper()} EP 1] Tick {ticks_run:3} | {action_name:25} | "
                    f"Reward {reward:+.2f} | Total {total_reward:.2f} | Eps {agent.epsilon:.3f}"
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
            "crashed": getattr(env.env, "crashed", False),
        }
        all_session_logs.append(session_log)
        all_tick_logs.extend(logs)

        print(
            f"\n[✓] {current_task.upper()} Episode 1 Score: {total_reward:.2f} | "
            f"Final Env Score: {final_score:.3f}\n" + "-" * 50
        )

    return all_tick_logs, all_session_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural DQN agent for NeuralPagedAttention")
    parser.add_argument("--train", action="store_true", help="Run offline training")
    parser.add_argument("--task", type=str, default=None, help="Task for run_sim only")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes")
    parser.add_argument("--eval-every", type=int, default=40, help="Eval every N episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Shorter episodes + smaller curriculum for smoke tests")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    if args.train:
        train_offline(
            episodes=args.episodes,
            eval_every=args.eval_every,
            seed=args.seed,
            fast=args.fast,
            resume=args.resume,
        )
    else:
        run_sim(task=args.task)
