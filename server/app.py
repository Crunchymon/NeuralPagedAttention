from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import KVCacheAction, KVCacheObservation, KVCacheState, StepResult
from server.environment import KVCacheEnvironment

app = FastAPI(title="Neural PagedAttention", version="1.0.0")
env = KVCacheEnvironment()


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


class StepRequest(BaseModel):
    action_id: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    task = (req.task if req and req.task else "easy")
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    obs = env.reset(task=task)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {"task": task},
    }


@app.post("/step")
def step(req: StepRequest):
    if not (0 <= req.action_id <= 17):
        raise HTTPException(status_code=400, detail="action_id must be 0-17")
    obs, reward, done, info = env.step(req.action_id)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()
