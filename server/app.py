"""
app.py
------
FastAPI application — session-based NeuralPagedAttention benchmark API.

Architecture
------------
- All environment state lives inside isolated AgentSession objects.
- No global mutable env singleton — concurrent agents are fully independent.
- Sessions expire after NPA_INACTIVITY_TTL seconds of silence (default 10 min).
  Every /reset, /step, /state, /render, /score call refreshes the timer.
- A background asyncio task sweeps expired sessions every NPA_CLEANUP_INTERVAL.

HuggingFace Spaces deployment
------------------------------
- Run with exactly ONE worker:  uvicorn server.app:app --host 0.0.0.0 --port 7860
- The Dockerfile already sets port 7860 (HF requirement).
- In-memory session store is safe for single-worker deployments.
- All config is read from environment variables (.env → HF Secrets).

Start locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from server.env_components.constants import (
    INACTIVITY_TTL,
    MAX_SESSIONS,
    SESSION_TTL_DEFAULT,
)
from server.session_manager import SessionManager

# ── Singletons ────────────────────────────────────────────────────────────────

sessions = SessionManager()
_start_time = time.time()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background cleanup task on startup; cancel it on shutdown."""
    cleanup_task = asyncio.create_task(sessions.cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuralPagedAttention Benchmark API",
    version="2.0.0",
    description=(
        "Session-based RL benchmark for KV-cache eviction policy evaluation. "
        "Each agent creates its own isolated session. "
        "Sessions expire after inactivity (default 10 min). "
        "See openapi.yaml for full contract."
    ),
    lifespan=lifespan,
    # Expose the full OpenAPI JSON at /openapi.json (needed by HF Spaces auto-docs)
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    agent_id: Optional[str] = Field(
        None,
        description="Human-readable agent identifier for leaderboard attribution.",
        example="ppo-agent-v3",
    )
    task: Optional[str] = Field(
        "easy",
        description="Initial task difficulty. One of: easy, medium, hard.",
        example="easy",
    )
    inactivity_ttl_seconds: int = Field(
        INACTIVITY_TTL,
        ge=60,
        le=86400,
        description=(
            "Seconds of inactivity before the session auto-expires. "
            "The timer resets on every /reset, /step, /state, /render, /score call. "
            f"Default: {INACTIVITY_TTL}s (10 min)."
        ),
    )


class ResetRequest(BaseModel):
    task: Optional[str] = Field(
        None,
        description="Override task difficulty for this episode. One of: easy, medium, hard.",
    )


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=17, description="Discrete action ID (0–17).")
    # Legacy alias so old agents using action_id= still work
    action_id: Optional[int] = Field(None, ge=0, le=17, exclude=True)

    def resolved_action(self) -> int:
        return self.action if self.action is not None else self.action_id


# ── Health / Info ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"], summary="Server health check")
def health_check():
    """
    Returns server status, active session count, and uptime.
    Useful for HuggingFace Spaces liveness probes.
    """
    active = len([s for s in sessions._sessions.values() if not s.is_expired])
    return {
        "status": "ok",
        "active_sessions": active,
        "max_sessions": MAX_SESSIONS,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "inactivity_ttl_default_seconds": INACTIVITY_TTL,
    }


@app.get("/info", tags=["health"], summary="Environment metadata")
def env_info():
    """
    Returns environment metadata so agents can self-configure.
    Mirrors the x-environment-config block in openapi.yaml.
    """
    from server.env_components.constants import (
        CRASH_PENALTY,
        PHASE_CONFIGS,
        SLA_MISS_FREE,
        SLA_MISS_VIP,
        ADMIT_BONUS,
        FREE_QUEUE_MAX,
        VIP_QUEUE_MAX,
        GPU_TOTAL_BLOCKS,
        CPU_TOTAL_BLOCKS,
        INACTIVITY_TTL,
        MAX_SESSIONS,
    )
    return {
        "observation_dim": 20,
        "action_space_size": 18,
        "action_descriptions": {
            "0":  "evict_largest_free — evict largest idle Free-tier block from GPU",
            "1":  "evict_largest_vip  — evict largest idle VIP-tier block from GPU",
            "2":  "evict_oldest_free  — evict oldest idle Free-tier block from GPU",
            "3":  "evict_oldest_vip   — evict oldest idle VIP-tier block from GPU",
            "4":  "swap_largest_free_to_cpu — swap largest idle Free block GPU→CPU",
            "5":  "swap_largest_vip_to_cpu  — swap largest idle VIP block GPU→CPU",
            "6":  "swap_oldest_free_to_cpu  — swap oldest idle Free block GPU→CPU",
            "7":  "swap_oldest_vip_to_cpu   — swap oldest idle VIP block GPU→CPU",
            "8":  "admit_next_free — admit next Free request from queue to GPU",
            "9":  "admit_next_vip  — admit next VIP request from queue to GPU",
            "10": "reject_next_free — drop next Free request from queue",
            "11": "reject_next_vip  — drop next VIP request from queue",
            "12": "preempt_shred_free — interrupt & permanently delete largest active Free",
            "13": "preempt_shred_vip  — interrupt & permanently delete largest active VIP",
            "14": "preempt_swap_free  — interrupt active Free and move to CPU",
            "15": "preempt_swap_vip   — interrupt active VIP and move to CPU",
            "16": "garbage_collect — delete Free CPU caches idle > 200 ticks",
            "17": "do_nothing — take no action this tick",
        },
        "phase_configs": PHASE_CONFIGS,
        "hardware": {
            "gpu_total_blocks": GPU_TOTAL_BLOCKS,
            "cpu_total_blocks": CPU_TOTAL_BLOCKS,
        },
        "queue_limits": {
            "free_queue_max": FREE_QUEUE_MAX,
            "vip_queue_max": VIP_QUEUE_MAX,
        },
        "reward_bounds": {"min": -100.0, "max": 3.0},
        "reward_shaping": {
            "crash_penalty": CRASH_PENALTY,
            "sla_miss_free": SLA_MISS_FREE,
            "sla_miss_vip": SLA_MISS_VIP,
            "admit_bonus": ADMIT_BONUS,
        },
        "session": {
            "max_sessions": MAX_SESSIONS,
            "default_inactivity_ttl_seconds": INACTIVITY_TTL,
        },
    }


@app.get("/metadata", tags=["health"], summary="OpenEnv-compatible metadata")
def metadata():
    """Backwards-compatible OpenEnv manifest (kept for existing test agents)."""
    return {
        "name": "neural-paged-attention",
        "description": (
            "RL environment simulating GPU KV Cache management "
            "for LLM inference servers."
        ),
        "tasks": ["easy", "medium", "hard"],
        "version": "2.0.0",
        "observation_space": {"type": "array", "shape": [20], "dtype": "float32"},
        "action_space": {"type": "discrete", "n": 18},
        "reward_range": [-100.0, 3.0],
    }


@app.get("/schema", tags=["health"], summary="Pydantic JSON schemas for all models")
def schema():
    from models import KVCacheAction, KVCacheObservation, KVCacheState
    return {
        "action":      KVCacheAction.model_json_schema(),
        "observation": KVCacheObservation.model_json_schema(),
        "state":       KVCacheState.model_json_schema(),
    }


# ── Session lifecycle ──────────────────────────────────────────────────────────

@app.post(
    "/sessions",
    status_code=201,
    tags=["sessions"],
    summary="Create a new isolated agent session",
)
def create_session(req: CreateSessionRequest = CreateSessionRequest()):
    """
    Allocate an isolated environment instance.
    Returns a ``session_id`` (UUID) that must be passed to all subsequent calls.

    The inactivity timer starts immediately.  Every call to /reset, /step, /state,
    /render, or /score resets the timer.  When the timer exceeds
    ``inactivity_ttl_seconds``, the session is automatically cleaned up.
    """
    task = req.task or "easy"
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail={"error": "invalid_task", "detail": f"Unknown task: {task}"})

    session = sessions.create(
        agent_id=req.agent_id,
        task_override=task,
        inactivity_ttl=req.inactivity_ttl_seconds,
    )
    return {
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "created_at": session.created_at.isoformat(),
        "expires_after_inactivity_seconds": session.inactivity_ttl,
        "status": session.status,
        "message": (
            f"Session created. Call POST /sessions/{session.session_id}/reset to start an episode. "
            f"Session expires after {session.inactivity_ttl}s of inactivity."
        ),
    }


@app.get("/sessions", tags=["sessions"], summary="List all active sessions")
def list_sessions():
    """Returns summaries of all non-expired sessions."""
    all_sessions = sessions.list_all()
    return {"sessions": all_sessions, "total": len(all_sessions)}


@app.get(
    "/sessions/{session_id}",
    tags=["sessions"],
    summary="Get session metadata",
)
def get_session(session_id: str = Path(..., description="Session UUID")):
    session = sessions.get(session_id)
    return session.to_summary()


@app.delete(
    "/sessions/{session_id}",
    tags=["sessions"],
    summary="Terminate and clean up a session",
)
def delete_session(session_id: str = Path(..., description="Session UUID")):
    return sessions.delete(session_id)


# ── Environment interaction ────────────────────────────────────────────────────

@app.post(
    "/sessions/{session_id}/reset",
    tags=["environment"],
    summary="Reset environment — start a new episode",
)
def reset_env(
    session_id: str = Path(..., description="Session UUID"),
    req: ResetRequest = ResetRequest(),
):
    """
    Resets state, queues, and reward accumulators for a new episode.
    Returns the initial observation.  Refreshes the inactivity timer.
    """
    session = sessions.get(session_id)  # touches session
    env = session.env

    task = req.task or env.task or "easy"
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail={"error": "invalid_task", "detail": f"Unknown task: {task}"})

    obs = env.reset(task=task)
    session.status = "running"

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {
            "task": env.task,
            "episode_id": env.episode_id,
            "tick": 0,
            "max_ticks": env.config["max_ticks"],
        },
    }


@app.post(
    "/sessions/{session_id}/step",
    tags=["environment"],
    summary="Apply an action and advance one timestep",
)
def step_env(
    req: StepRequest,
    session_id: str = Path(..., description="Session UUID"),
):
    """
    Executes an action, advances the tick engine, generates new traffic,
    computes reward, and returns the next observation.
    Call /reset before the first /step.  Refreshes the inactivity timer.
    """
    session = sessions.get(session_id)  # touches session

    if session.status == "idle":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "episode_not_started",
                "detail": f"Call POST /sessions/{session_id}/reset before stepping.",
            },
        )

    # Support both action= (new) and action_id= (legacy)
    action = req.action if req.action is not None else req.action_id
    if action is None or not (0 <= action <= 17):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_action", "detail": "action must be an integer 0–17."},
        )

    env = session.env
    obs, reward, done, info = env.step(action)

    if done:
        session.status = "idle"
        session.episodes_completed += 1

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get(
    "/sessions/{session_id}/state",
    tags=["environment"],
    summary="Poll current state without stepping",
)
def get_state(session_id: str = Path(..., description="Session UUID")):
    """
    Returns the current observation and episode metadata without advancing
    the simulation.  Safe to call multiple times.  Refreshes inactivity timer.
    """
    session = sessions.get(session_id)  # touches session
    env = session.env
    obs = env._build_observation()
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "state": env.state().model_dump(),
        "gpu_util": env.ledger.gpu_utilization(),
        "cpu_util": env.ledger.cpu_utilization(),
        "free_queue_len": len(env.free_queue),
        "vip_queue_len": len(env.vip_queue),
    }


@app.get(
    "/sessions/{session_id}/render",
    tags=["environment"],
    summary="Human-readable text rendering of current state",
)
def render_state(session_id: str = Path(..., description="Session UUID")):
    """
    Returns a structured text description of the current state, suitable for
    feeding directly into an LLM agent as context.  Refreshes inactivity timer.
    """
    session = sessions.get(session_id)  # touches session
    env = session.env
    obs = env._build_observation()
    st = env.state()

    text = (
        f"Tick {st.tick}/{st.max_ticks}. "
        f"Task: {st.task.upper()}. "
        f"GPU: {env.ledger.gpu_utilization():.0%} full. "
        f"CPU: {env.ledger.cpu_utilization():.0%} full. "
        f"Free queue: {len(env.free_queue)} requests. "
        f"VIP queue: {len(env.vip_queue)} requests. "
        f"Completed: {st.total_completed}/{st.total_arrived}. "
        f"Score: {st.current_score:.4f}. "
        f"Last action: {env._last_action_result}."
    )
    return {"session_id": session_id, "text": text, "structured": obs.model_dump()}


# ── Scoring ────────────────────────────────────────────────────────────────────

@app.get(
    "/sessions/{session_id}/score",
    tags=["benchmark"],
    summary="Get current episode score breakdown",
)
def get_score(session_id: str = Path(..., description="Session UUID")):
    """Returns live score metrics for the current episode.  Refreshes inactivity timer."""
    session = sessions.get(session_id)  # touches session
    env = session.env
    st = env.state()
    return {
        "session_id": session_id,
        "episode_id": st.episode_id,
        "task": st.task,
        "tick": st.tick,
        "max_ticks": st.max_ticks,
        "cumulative_reward": st.cumulative_reward,
        "current_score": st.current_score,
        "total_arrived": st.total_arrived,
        "total_completed": st.total_completed,
        "total_rejected": st.total_rejected,
        "crashed": st.total_crashed,
        "gpu_utilization": env.ledger.gpu_utilization(),
        "cpu_utilization": env.ledger.cpu_utilization(),
    }


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.get(
    "/sessions/{session_id}/dashboard",
    tags=["benchmark"],
    summary="Per-session live dashboard snapshot",
)
def session_dashboard(session_id: str = Path(..., description="Session UUID")):
    """
    Detailed per-session snapshot for dashboard visualisation.
    Includes environment state, observation vector, memory layout, and queue stats.
    Refreshes the inactivity timer.
    """
    session = sessions.get(session_id)  # touches session
    env = session.env
    obs = env._build_observation()
    st = env.state()
    return {
        "session_id": session_id,
        "agent_id": session.agent_id,
        "status": session.status,
        "episodes_completed": session.episodes_completed,
        "expires_in_seconds": round(
            session.inactivity_ttl - session.inactive_seconds, 1
        ),
        "state": st.model_dump(),
        "observation": obs.model_dump(),
        "memory": {
            "gpu_util": env.ledger.gpu_utilization(),
            "cpu_util": env.ledger.cpu_utilization(),
            "gpu_used_blocks": env.ledger.gpu_used,
            "cpu_used_blocks": env.ledger.cpu_used,
        },
        "queues": {
            "free_queue_len": len(env.free_queue),
            "vip_queue_len": len(env.vip_queue),
        },
        "last_action_result": env._last_action_result,
    }


@app.get(
    "/dashboard",
    tags=["benchmark"],
    summary="Global aggregate dashboard across all sessions",
)
def global_dashboard():
    """
    Aggregate view across all active sessions.
    Useful for monitoring dashboards and server-level health visualisation.
    """
    return sessions.aggregate_dashboard()


# ── Legacy routes (deprecated, kept for backwards compatibility) ───────────────

_DEPRECATION_MSG = (
    "This endpoint is deprecated. Use the session-based API instead: "
    "POST /sessions → POST /sessions/{session_id}/reset → POST /sessions/{session_id}/step"
)

_DEFAULT_TASK = "easy"
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
        headers={"Deprecation": "true", "Link": "/docs#tag/sessions; rel='successor-version'"},
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


# ── MCP endpoint (unchanged) ───────────────────────────────────────────────────

from fastapi import Request as FastAPIRequest


@app.post("/mcp", tags=["health"], summary="MCP capability handshake")
async def mcp(request: FastAPIRequest):
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
            "description": "KV Cache GPU memory manager (session-based v2)",
            "capabilities": {
                "sessions": True,
                "reset": True,
                "step": True,
                "state": True,
                "dashboard": True,
            },
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    # HuggingFace Spaces: always --workers 1 with in-memory session store
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1, reload=False)


if __name__ == "__main__":
    main()