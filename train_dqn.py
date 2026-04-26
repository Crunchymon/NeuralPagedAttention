#!/usr/bin/env python3
"""
Train the DQN agent without relying on ``dqn.py``'s CLI block.

Use this on Colab if an older fork still does ``run_sim(task=sys.argv[1])`` and
breaks on ``python agents/NeuralAgent/dqn.py --train``.

Run from repo root:

  python train_dqn.py --episodes 1000 --eval-every 50 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dqn_module():
    path = os.path.join(ROOT, "agents", "NeuralAgent", "dqn.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("npa_dqn_train", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    sys.path.insert(0, ROOT)
    parser = argparse.ArgumentParser(description="DQN training entry (Colab-safe)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    dqn = _load_dqn_module()
    if not hasattr(dqn, "train_offline"):
        print(
            "[!] Your agents/NeuralAgent/dqn.py is too old (no train_offline). "
            "Pull the latest NeuralPagedAttention from GitHub.",
            file=sys.stderr,
        )
        sys.exit(1)

    dqn.train_offline(
        episodes=args.episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        fast=args.fast,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
