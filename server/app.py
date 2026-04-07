from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import KVCacheAction, KVCacheObservation, KVCacheState, StepResult
from server.environment import KVCacheEnvironment

app = FastAPI(title="Neural PagedAttention", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = KVCacheEnvironment()


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


class StepRequest(BaseModel):
    action: Optional[int] = None       # OpenEnv standard field
    action_id: Optional[int] = None    # Legacy internal field


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
    action = req.action if req.action is not None else req.action_id
    if action is None or not (0 <= action <= 17):
        raise HTTPException(status_code=400, detail="action must be 0-17")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/metadata")
def metadata():
    return {
        "name": "neural-paged-attention",
        "description": "RL environment simulating GPU KV Cache management for LLM inference servers.",
        "tasks": ["easy", "medium", "hard"],
        "version": "1.0.0",
        "observation_space": {"type": "array", "shape": [20], "dtype": "float32"},
        "action_space": {"type": "discrete", "n": 18},
        "reward_range": [-100.0, 3.0],
    }


@app.get("/schema")
def schema():
    from models import KVCacheAction, KVCacheObservation, KVCacheState
    return {
        "action":      KVCacheAction.model_json_schema(),
        "observation": KVCacheObservation.model_json_schema(),
        "state":       KVCacheState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    req_id = body.get("id") if isinstance(body, dict) else None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "neural-paged-attention",
            "description": "KV Cache GPU memory manager",
            "capabilities": {"reset": True, "step": True, "state": True},
        },
    }

@app.get("/dashboard")
def dashboard():
    return {
        "state": env.state().model_dump(),
        "gpu_util": env.ledger.gpu_utilization(),
        "cpu_util": env.ledger.cpu_utilization(),
        "free_queue": len(env.free_queue),
        "vip_queue": len(env.vip_queue),
        "last_action_result": env._last_action_result,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()