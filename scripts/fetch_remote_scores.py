"""
Fetch final scores for the LLM and PPO agents by calling the public HF Space
backend (the same backend the dashboard uses). Writes the augmented results
back to results/benchmarks.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "benchmarks.json"
BASE = "https://suryanshchattree-neural-paged-attention-env.hf.space"

REMOTE_AGENTS = ["llm", "ppo"]
TASKS = ["easy", "medium", "hard"]
TIMEOUT_S = 60 * 60  # 1h per request — LLM/PPO on CPU Spaces is slow

OUT_PER_RUN = REPO / "results" / "remote_runs"


def run_one(agent: str, task: str) -> dict:
    print(f">>> remote {agent}/{task} ...", flush=True)
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/api/simulate",
            json={"agent": agent, "task": task},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    ERROR: {exc}  ({elapsed:.1f}s)", flush=True)
        return {"agent": agent, "task": task, "ok": False, "error": str(exc),
                "elapsed_s": round(elapsed, 1), "remote": True}

    elapsed = time.time() - t0
    sessions = data.get("session_logs") or []
    if not sessions:
        return {"agent": agent, "task": task, "ok": False,
                "error": "no session logs", "elapsed_s": round(elapsed, 1),
                "remote": True}

    OUT_PER_RUN.mkdir(parents=True, exist_ok=True)
    (OUT_PER_RUN / f"{agent}_{task}.json").write_text(json.dumps(data))

    s = sessions[-1]
    out = {
        "agent": agent, "task": task, "ok": True, "remote": True,
        "final_score": float(s.get("final_score", 0.0)),
        "total_reward": float(s.get("total_reward", 0.0)),
        "ticks_run": int(s.get("ticks_run", 0)),
        "total_arrived": int(s.get("total_arrived", 0)),
        "total_completed": int(s.get("total_completed", 0)),
        "crashed": bool(s.get("crashed", False)),
        "elapsed_s": round(elapsed, 1),
    }
    print(
        f"    score={out['final_score']:.3f}  "
        f"completed={out['total_completed']}/{out['total_arrived']}  "
        f"ticks={out['ticks_run']}  ({elapsed:.1f}s)",
        flush=True,
    )
    return out


def main() -> int:
    bench = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"results": [], "table": {}}
    results: list[dict] = list(bench.get("results", []))
    table: dict[str, dict] = dict(bench.get("table", {}))

    for agent in REMOTE_AGENTS:
        table.setdefault(agent, {})
        for task in TASKS:
            res = run_one(agent, task)
            results.append(res)
            table[agent][task] = res.get("final_score") if res.get("ok") else None
            with RESULTS.open("w") as f:
                json.dump({"results": results, "table": table}, f, indent=2)

    print("\nFull score table:\n")
    print(f"{'agent':<12} {'easy':>8} {'medium':>8} {'hard':>8}")
    for a, row in table.items():
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "ERR"
        print(f"{a:<12} {fmt(row.get('easy')):>8} {fmt(row.get('medium')):>8} {fmt(row.get('hard')):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
