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

app = FastAPI(title="Neural PagedAttention Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    agent: str = Field(..., description="The name of the agent to run. Must be either 'lru' or 'random'.")
    task: Optional[str] = Field(None, description="The task difficulty to simulate. Options: 'easy', 'medium', 'hard'. If omitted, all tasks will be run sequentially.")

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
        if agent_type == "lru":
            tick_logs, session_logs = run_lru(task=req.task)
        elif agent_type == "random":
            tick_logs, session_logs = run_random(task=req.task)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {req.agent}. Supported: 'lru', 'random'.")
            
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
