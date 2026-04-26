#!/usr/bin/env python3
"""
Marathon training launcher for Google Colab / cloud GPU.

Run from the **repository root** after installing dependencies (see Colab notebook).

  python colab/marathon_train.py --dqn-only
  python colab/marathon_train.py --ppo-only
  python colab/marathon_train.py

Defaults favor quality over speed (long runs).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-only", action="store_true")
    parser.add_argument("--ppo-only", action="store_true")
    parser.add_argument("--dqn-episodes", type=int, default=1000)
    parser.add_argument("--dqn-eval-every", type=int, default=50)
    parser.add_argument("--ppo-episodes", type=int, default=220)
    parser.add_argument("--ppo-max-ticks", type=int, default=0, help="0 = full task horizon")
    parser.add_argument("--ppo-lr", type=float, default=2e-5)
    parser.add_argument("--ppo-entropy", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = repo_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    py = sys.executable

    run_dqn = not args.ppo_only
    run_ppo = not args.dqn_only

    if args.dqn_only and args.ppo_only:
        print("Use only one of --dqn-only / --ppo-only, or neither for both.")
        sys.exit(1)

    if run_dqn:
        print("=" * 70)
        print("MARATHON DQN")
        print("=" * 70)
        cmd = [
            py,
            os.path.join(root, "agents", "NeuralAgent", "dqn.py"),
            "--train",
            "--episodes",
            str(args.dqn_episodes),
            "--eval-every",
            str(args.dqn_eval_every),
            "--seed",
            str(args.seed),
        ]
        subprocess.check_call(cmd)

    if run_ppo:
        print("=" * 70)
        print("MARATHON LLM POLICY GRADIENT (Unsloth / LoRA)")
        print("=" * 70)
        cmd = [
            py,
            os.path.join(root, "agents", "UnslothPPOAgent", "train_ppo.py"),
            "--episodes",
            str(args.ppo_episodes),
            "--lr",
            str(args.ppo_lr),
            "--entropy-coef",
            str(args.ppo_entropy),
        ]
        if args.ppo_max_ticks > 0:
            cmd.extend(["--max-ticks", str(args.ppo_max_ticks)])
        subprocess.check_call(cmd)

    print("\nArtifacts:")
    print(f"  DQN last:     {os.path.join(root, 'agents', 'NeuralAgent', 'dqn_weights.pth')}")
    print(f"  DQN best:     {os.path.join(root, 'agents', 'NeuralAgent', 'dqn_weights_best.pth')}")
    print(f"  PPO LoRA:     {os.path.join(root, 'agents', 'UnslothPPOAgent', 'ppo_lora_agent')}")


if __name__ == "__main__":
    main()
