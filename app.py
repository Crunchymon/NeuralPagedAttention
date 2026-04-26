import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(__file__))

from agents.LRUAgent.lru import run_sim as run_lru
from agents.RandomAgent.run_random_agent import run_simulation as run_random
from agents.QLearningAgent.QAgent import run_sim as run_qlearning
from agents.NeuralAgent.dqn import run_sim as run_dqn
from agents.LLMAgent.llm import run_sim as run_llm

import server.env_components.constants as constants
from server.env_components.traffic import build_traffic_trace

app = FastAPI(title="Neural PagedAttention Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    agent: str = Field(..., description="The name of the agent to run. Must be either 'lru', 'random', 'qlearning', or 'neural'.")
    task: Optional[str] = Field(None, description="The task difficulty to simulate. Options: 'easy', 'medium', 'hard'. If omitted, all tasks will be run sequentially.")
    ticks: Optional[int] = Field(None, description="Optional override for the maximum number of ticks to run for the simulation. If omitted, default task duration is used.")

class SettingsRequest(BaseModel):
    gpu_total_blocks: Optional[int] = None
    cpu_total_blocks: Optional[int] = None
    tokens_per_block: Optional[int] = None
    max_queue_size: Optional[int] = None
    max_ticks_easy: Optional[int] = None
    max_ticks_medium: Optional[int] = None
    max_ticks_hard: Optional[int] = None


class BatchSimulationRequest(BaseModel):
    gpu_total_blocks: Optional[int] = None
    cpu_total_blocks: Optional[int] = None
    tokens_per_block: Optional[int] = None
    max_queue_size: Optional[int] = None
    max_ticks_easy: Optional[int] = None
    max_ticks_medium: Optional[int] = None
    max_ticks_hard: Optional[int] = None
    seed: int = Field(..., ge=0)
    agents: Optional[list[str]] = None
    tasks: Optional[list[str]] = None
    include_tick_logs: bool = True


AGENT_RUNNERS = {
    "lru": run_lru,
    "random": run_random,
    "qlearning": run_qlearning,
    "neural": run_dqn,
    "llm": run_llm,
}


def _apply_runtime_settings(
    gpu_total_blocks: Optional[int] = None,
    cpu_total_blocks: Optional[int] = None,
    tokens_per_block: Optional[int] = None,
    max_queue_size: Optional[int] = None,
    max_ticks_easy: Optional[int] = None,
    max_ticks_medium: Optional[int] = None,
    max_ticks_hard: Optional[int] = None,
):
    if gpu_total_blocks is not None:
        constants.GPU_TOTAL_BLOCKS = gpu_total_blocks
    if cpu_total_blocks is not None:
        constants.CPU_TOTAL_BLOCKS = cpu_total_blocks
    if tokens_per_block is not None:
        constants.TOKENS_PER_BLOCK = tokens_per_block
    if max_queue_size is not None:
        constants.FREE_QUEUE_MAX = max_queue_size
        constants.VIP_QUEUE_MAX = max_queue_size

    if max_ticks_easy is not None:
        constants.PHASE_CONFIGS["easy"]["max_ticks"] = max_ticks_easy
    if max_ticks_medium is not None:
        constants.PHASE_CONFIGS["medium"]["max_ticks"] = max_ticks_medium
    if max_ticks_hard is not None:
        constants.PHASE_CONFIGS["hard"]["max_ticks"] = max_ticks_hard


def _runtime_settings_snapshot() -> dict:
    return {
        "GPU_TOTAL_BLOCKS": constants.GPU_TOTAL_BLOCKS,
        "CPU_TOTAL_BLOCKS": constants.CPU_TOTAL_BLOCKS,
        "TOKENS_PER_BLOCK": constants.TOKENS_PER_BLOCK,
        "FREE_QUEUE_MAX": constants.FREE_QUEUE_MAX,
        "VIP_QUEUE_MAX": constants.VIP_QUEUE_MAX,
        "PHASE_CONFIGS": constants.PHASE_CONFIGS,
    }


def _run_agent(agent_type: str, task: str, ticks: Optional[int], seed: int, traffic_trace):
    runner = AGENT_RUNNERS.get(agent_type)
    if runner is None:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_type}. Supported: {sorted(AGENT_RUNNERS)}")
    return runner(task=task, ticks=ticks, seed=seed, traffic_trace=traffic_trace)

@app.get(
    "/api/agents",
    tags=["simulation"],
    summary="List available agents",
    description="Returns a list of all supported reinforcement learning and heuristic agents, along with instructions on how to use them."
)
def get_agents():
    return {
        "status": "success",
        "available_agents": [
            {"id": "lru", "name": "Least Recently Used (LRU)", "description": "Heuristic agent that evicts the least recently used cache."},
            {"id": "random", "name": "Random Agent", "description": "Baseline agent that selects actions entirely at random."},
            {"id": "qlearning", "name": "Tabular Q-Learning Agent", "description": "RL agent using a discretized state-space Q-table."},
            {"id": "neural",    "name": "Deep Q-Network (DQN) Agent", "description": "RL agent using a PyTorch neural network to map continuous states to Q-values."},
            {"id": "llm",       "name": "Qwen2.5-3B LLM Agent",       "description": "Locally-running Qwen2.5-3B-Instruct model that chooses actions from natural-language prompts. Auto-quantized (4-bit on CPU, bfloat16 on GPU)."}
        ],
        "usage": {
            "endpoint": "POST /api/simulate",
            "example_payload": {
                "agent": "neural",
                "task": "hard"
            },
            "description": "Send a POST request to /api/simulate with the desired agent ID and task difficulty ('easy', 'medium', or 'hard')."
        }
    }

@app.get(
    "/api/settings",
    tags=["configuration"],
    summary="Get environment constants",
    description="Returns the current dynamic environment constants and their original default values."
)
def get_settings():
    return {
        "status": "success",
        "current_settings": _runtime_settings_snapshot(),
        "default_settings": {
            "GPU_TOTAL_BLOCKS": 1000,
            "CPU_TOTAL_BLOCKS": 5000,
            "TOKENS_PER_BLOCK": 16,
            "max_queue_size": 100,
            "max_ticks_easy": 2000,
            "max_ticks_medium": 5000,
            "max_ticks_hard": 10000
        }
    }

@app.post(
    "/api/settings",
    tags=["configuration"],
    summary="Update environment constants",
    description="Update GPU_TOTAL_BLOCKS, CPU_TOTAL_BLOCKS, TOKENS_PER_BLOCK, and max_ticks for tasks."
)
def update_settings(req: SettingsRequest):
    _apply_runtime_settings(
        gpu_total_blocks=req.gpu_total_blocks,
        cpu_total_blocks=req.cpu_total_blocks,
        tokens_per_block=req.tokens_per_block,
        max_queue_size=req.max_queue_size,
        max_ticks_easy=req.max_ticks_easy,
        max_ticks_medium=req.max_ticks_medium,
        max_ticks_hard=req.max_ticks_hard,
    )

    return {
        "status": "success",
        "current_settings": _runtime_settings_snapshot(),
        "default_settings": {
            "GPU_TOTAL_BLOCKS": 1000,
            "CPU_TOTAL_BLOCKS": 5000,
            "TOKENS_PER_BLOCK": 16,
            "max_queue_size": 100,
            "max_ticks_easy": 2000,
            "max_ticks_medium": 3000,
            "max_ticks_hard": 5000
        }
    }   


@app.post(
    "/api/simulate-all",
    tags=["simulation"],
    summary="Run all agents and tasks with shared seeded traffic",
    description="Applies runtime configuration, generates one deterministic traffic trace per task from the provided seed, runs every selected agent, and returns combined tick and session telemetry.",
)
def run_all_simulations(req: BatchSimulationRequest):
    _apply_runtime_settings(
        gpu_total_blocks=req.gpu_total_blocks,
        cpu_total_blocks=req.cpu_total_blocks,
        tokens_per_block=req.tokens_per_block,
        max_queue_size=req.max_queue_size,
        max_ticks_easy=req.max_ticks_easy,
        max_ticks_medium=req.max_ticks_medium,
        max_ticks_hard=req.max_ticks_hard,
    )

    selected_agents = [agent.lower() for agent in (req.agents or list(AGENT_RUNNERS.keys()))]
    selected_tasks = [task.lower() for task in (req.tasks or ["easy", "medium", "hard"])]

    for agent in selected_agents:
        if agent not in AGENT_RUNNERS:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent}. Supported: {sorted(AGENT_RUNNERS)}")
    for task in selected_tasks:
        if task not in constants.PHASE_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Unknown task: {task}. Supported: {sorted(constants.PHASE_CONFIGS)}")

    results_by_task: dict[str, dict[str, dict]] = {}
    all_session_logs: list[dict] = []
    all_tick_logs: list[dict] = []

    for task in selected_tasks:
        task_config = constants.PHASE_CONFIGS[task]
        traffic_trace = build_traffic_trace(
            task=task_config["traffic_fn"],
            max_ticks=task_config["max_ticks"],
            seed=req.seed,
            vip_ratio=task_config["vip_ratio"],
            power_user_pct=task_config["power_user_pct"],
        )
        results_by_task[task] = {}

        for agent in selected_agents:
            tick_logs, session_logs = _run_agent(
                agent_type=agent,
                task=task,
                ticks=task_config["max_ticks"],
                seed=req.seed,
                traffic_trace=traffic_trace,
            )

            enriched_session_logs = []
            for session_log in session_logs:
                session_copy = dict(session_log)
                session_copy["agent"] = agent
                session_copy["task"] = task
                enriched_session_logs.append(session_copy)
                all_session_logs.append(session_copy)

            enriched_tick_logs = []
            if req.include_tick_logs:
                for tick_log in tick_logs:
                    tick_copy = dict(tick_log)
                    tick_copy["agent"] = agent
                    tick_copy["task"] = task
                    enriched_tick_logs.append(tick_copy)
                    all_tick_logs.append(tick_copy)

            results_by_task[task][agent] = {
                "session_logs": enriched_session_logs,
                "tick_logs": enriched_tick_logs if req.include_tick_logs else [],
            }

    return {
        "status": "success",
        "seed": req.seed,
        "applied_config": _runtime_settings_snapshot(),
        "tasks": selected_tasks,
        "agents": selected_agents,
        "results_by_task": results_by_task,
        "all_session_logs": all_session_logs,
        "all_tick_logs": all_tick_logs if req.include_tick_logs else [],
    }

@app.post(
    "/api/simulate",
    tags=["simulation"],
    summary="Run a full batch simulation",
    description="""
    Triggers a headless background simulation of the Neural PagedAttention environment using the specified agent policy.
    
    This endpoint executes the entire episode(s) and returns a comprehensive telemetry package containing:
    - `session_logs`: High-level episodic summaries including normalized scores and crash flags.
    - `tick_logs`: Granular, tick-by-tick telemetry including rewards, actions, and 20-dimensional observation vectors.
    
    Multiple simulations can be requested concurrently without interference.
    """
)
def run_simulation_endpoint(req: SimulationRequest):
    agent_type = req.agent.lower()
    
    try:
        tasks_to_run = [req.task] if req.task else ["easy", "medium", "hard"]
        tick_logs = []
        session_logs = []

        for task in tasks_to_run:
            task_tick_logs, task_session_logs = _run_agent(agent_type, task, req.ticks, seed=42, traffic_trace=None)
            tick_logs.extend(task_tick_logs)
            session_logs.extend(task_session_logs)
            
        return {
            "status": "success",
            "agent": agent_type,
            "task": req.task or "mixed",
            "session_logs": session_logs,
            "tick_logs": tick_logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- LEGACY ENDPOINTS ---------------- #

_legacy_env = None  # Lazily initialised global fallback env

def _get_legacy_env():
    """Return (or create) the single legacy global environment for old agents."""
    global _legacy_env
    if _legacy_env is None:
        from server.environment import KVCacheEnvironment
        _legacy_env = KVCacheEnvironment()
    return _legacy_env

class _LegacyResetRequest(BaseModel):
    task: Optional[str] = "easy"

class _LegacyStepRequest(BaseModel):
    action: Optional[int] = None
    action_id: Optional[int] = None

@app.post("/reset", tags=["legacy"], summary="[DEPRECATED] Reset global env", deprecated=True)
def legacy_reset(req: _LegacyResetRequest = _LegacyResetRequest()):
    """**Deprecated.** Use ``POST /sessions/{session_id}/reset`` instead."""
    env = _get_legacy_env()
    task = req.task or "easy"
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    obs = env.reset(task=task)
    return JSONResponse(
        content={"observation": obs.model_dump(), "reward": 0.0, "done": False, "info": {"task": task}},
        headers={"Deprecation": "true"},
    )

@app.post("/step", tags=["legacy"], summary="[DEPRECATED] Step global env", deprecated=True)
def legacy_step(req: _LegacyStepRequest):
    """**Deprecated.** Use ``POST /sessions/{session_id}/step`` instead."""
    env = _get_legacy_env()
    action = req.action if req.action is not None else req.action_id
    if action is None or not (0 <= action <= 17):
        raise HTTPException(status_code=400, detail="action must be 0-17")
    obs, reward, done, info = env.step(action)
    return JSONResponse(
        content={"observation": obs.model_dump(), "reward": reward, "done": done, "info": info},
        headers={"Deprecation": "true"},
    )

@app.get("/state", tags=["legacy"], summary="[DEPRECATED] Get global env state", deprecated=True)
def legacy_state():
    """**Deprecated.** Use ``GET /sessions/{session_id}/state`` instead."""
    env = _get_legacy_env()
    return JSONResponse(
        content=env.state().model_dump(),
        headers={"Deprecation": "true"},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
