"""
Benchmark runner: executes each agent on each task, records the env's final
score per session, and writes a JSON report to results/benchmarks.json.

Skips LLM and PPO agents by default (they require a 1.5B Qwen download).
Pass --include-llm and --include-ppo to opt in.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

AGENTS = [
    ("random",    "agents.RandomAgent.run_random_agent",   "run_simulation"),
    ("lru",       "agents.LRUAgent.lru",                   "run_sim"),
    ("qlearning", "agents.QLearningAgent.QAgent",          "run_sim"),
    ("neural",    "agents.NeuralAgent.dqn",                "run_sim"),
]

OPTIONAL_AGENTS = {
    "llm": ("agents.LLMAgent.llm",            "run_sim"),
    "ppo": ("agents.UnslothPPOAgent.ppo_agent", "run_sim"),
}

TASKS = ["easy", "medium", "hard"]


def import_run_sim(module_path: str, fn_name: str):
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def run_one(agent_id: str, module_path: str, fn_name: str, task: str) -> dict:
    print(f"\n>>> {agent_id} / {task} ...", flush=True)
    run_sim = import_run_sim(module_path, fn_name)

    t0 = time.time()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            tick_logs, session_logs = run_sim(task=task)
    except Exception as exc:
        print(f"    ERROR: {exc}", flush=True)
        traceback.print_exc()
        return {
            "agent": agent_id, "task": task, "ok": False,
            "error": str(exc), "elapsed_s": round(time.time() - t0, 2),
        }
    elapsed = time.time() - t0

    if not session_logs:
        return {
            "agent": agent_id, "task": task, "ok": False,
            "error": "no session logs returned",
            "elapsed_s": round(elapsed, 2),
        }

    s = session_logs[-1]
    result = {
        "agent": agent_id,
        "task": task,
        "ok": True,
        "final_score": float(s.get("final_score", 0.0)),
        "total_reward": float(s.get("total_reward", 0.0)),
        "ticks_run": int(s.get("ticks_run", 0)),
        "total_arrived": int(s.get("total_arrived", 0)),
        "total_completed": int(s.get("total_completed", 0)),
        "crashed": bool(s.get("crashed", False)),
        "elapsed_s": round(elapsed, 2),
    }
    print(
        f"    score={result['final_score']:.3f}  "
        f"completed={result['total_completed']}/{result['total_arrived']}  "
        f"ticks={result['ticks_run']}  ({elapsed:.1f}s)",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--include-ppo", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "benchmarks.json"))
    args = parser.parse_args()

    agents = list(AGENTS)
    if args.include_llm:
        agents.append(("llm", *OPTIONAL_AGENTS["llm"]))
    if args.include_ppo:
        agents.append(("ppo", *OPTIONAL_AGENTS["ppo"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for agent_id, module_path, fn_name in agents:
        for task in TASKS:
            results.append(run_one(agent_id, module_path, fn_name, task))

    table: dict[str, dict[str, float | None]] = {}
    for r in results:
        table.setdefault(r["agent"], {})[r["task"]] = r.get("final_score") if r.get("ok") else None

    out_path.write_text(json.dumps({"results": results, "table": table}, indent=2))
    print(f"\nWrote {out_path}")

    print("\nFinal score table:\n")
    print(f"{'agent':<10} {'easy':>8} {'medium':>8} {'hard':>8}")
    for agent_id in table:
        row = table[agent_id]
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "ERR"
        print(f"{agent_id:<10} {fmt(row.get('easy')):>8} {fmt(row.get('medium')):>8} {fmt(row.get('hard')):>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
