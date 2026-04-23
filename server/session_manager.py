"""
session_manager.py
------------------
Per-agent session management for the NeuralPagedAttention benchmark API.

Key design decisions
--------------------
- Each session owns its own ``KVCacheEnvironment`` instance — no shared mutable state.
- Sessions expire on **inactivity**, not on wall-clock age.  Every API call that
  touches a session (reset, step, state, render, score) refreshes ``last_active_at``.
  A session is only eligible for cleanup after ``inactivity_ttl`` seconds of silence.
- A background ``asyncio`` coroutine (``cleanup_loop``) runs every
  ``CLEANUP_INTERVAL`` seconds and evicts stale sessions.
- In-memory dict — safe for ``--workers 1`` (HuggingFace Spaces default).

Usage
-----
    sessions = SessionManager()
    # lifespan:
    task = asyncio.create_task(sessions.cleanup_loop())
    ...
    task.cancel()
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import HTTPException

from server.env_components.constants import (
    CLEANUP_INTERVAL,
    INACTIVITY_TTL,
    MAX_SESSIONS,
    SESSION_TTL_DEFAULT,
)
from server.environment import KVCacheEnvironment


# ── Session dataclass ──────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    """Holds all mutable state for a single connected agent."""

    session_id: str
    agent_id: Optional[str]
    env: KVCacheEnvironment
    created_at: datetime
    inactivity_ttl: int          # seconds of silence before expiry
    status: Literal["idle", "running"] = "idle"
    episodes_completed: int = 0
    _last_active_ts: float = field(default_factory=time.monotonic, repr=False)

    # ── Activity tracking ──────────────────────────────────────────────────

    def touch(self) -> None:
        """Call on every meaningful API interaction to reset the inactivity timer."""
        self._last_active_ts = time.monotonic()

    @property
    def inactive_seconds(self) -> float:
        return time.monotonic() - self._last_active_ts

    @property
    def is_expired(self) -> bool:
        return self.inactive_seconds > self.inactivity_ttl

    @property
    def expires_at(self) -> datetime:
        """Projected wall-clock time when session expires (based on current last-active)."""
        remaining = max(0.0, self.inactivity_ttl - self.inactive_seconds)
        return datetime.now(timezone.utc).replace(microsecond=0).__class__.fromtimestamp(
            time.time() + remaining, tz=timezone.utc
        )

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_summary(self) -> dict:
        env_state = self.env.state()
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "inactivity_ttl_seconds": self.inactivity_ttl,
            "inactive_for_seconds": round(self.inactive_seconds, 1),
            "status": self.status,
            "episodes_completed": self.episodes_completed,
            "current_task": env_state.task,
            "current_tick": env_state.tick,
        }


# ── Session Manager ────────────────────────────────────────────────────────────

class SessionManager:
    """
    Thread-safe (within a single asyncio event loop) in-memory session store.

    All public methods that modify ``_sessions`` are synchronous; they are safe
    because FastAPI runs a single-threaded asyncio loop with ``--workers 1``.
    """

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        cleanup_interval: int = CLEANUP_INTERVAL,
        default_inactivity_ttl: int = INACTIVITY_TTL,
    ) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self.max_sessions = max_sessions
        self.cleanup_interval = cleanup_interval
        self.default_inactivity_ttl = default_inactivity_ttl
        self._start_time = time.time()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def create(
        self,
        agent_id: Optional[str] = None,
        task_override: Optional[str] = None,
        inactivity_ttl: Optional[int] = None,
    ) -> AgentSession:
        """
        Allocate a new session.  Raises 429 if the session cap is reached.

        Parameters
        ----------
        agent_id:
            Human-readable identifier for leaderboard attribution.
        task_override:
            If provided, the environment is pre-seeded for this task on creation.
        inactivity_ttl:
            Override the default inactivity TTL for this session (seconds).
            Must be >= 60 and <= 86400.  Defaults to ``INACTIVITY_TTL`` (600 s).
        """
        if len(self._sessions) >= self.max_sessions:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "session_limit_reached",
                    "detail": (
                        f"Server capacity ({self.max_sessions} sessions) is full. "
                        "Delete an existing session or try again later."
                    ),
                },
            )

        ttl = inactivity_ttl if inactivity_ttl is not None else self.default_inactivity_ttl
        ttl = max(60, min(86400, ttl))

        env = KVCacheEnvironment()

        session = AgentSession(
            session_id=str(uuid.uuid4()),
            agent_id=agent_id,
            env=env,
            created_at=datetime.now(timezone.utc),
            inactivity_ttl=ttl,
        )

        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> AgentSession:
        """
        Retrieve a session by ID.

        Raises 404 if the session does not exist or has expired due to inactivity.
        On success, automatically refreshes the inactivity timer.
        """
        session = self._sessions.get(session_id)
        if session is None or session.is_expired:
            # Eagerly remove if expired
            self._sessions.pop(session_id, None)
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "session_not_found",
                    "detail": (
                        f"Session '{session_id}' does not exist or expired "
                        f"after {self.default_inactivity_ttl}s of inactivity."
                    ),
                },
            )
        session.touch()
        return session

    def delete(self, session_id: str) -> dict:
        """Explicitly terminate a session. Returns a summary dict."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "session_not_found",
                    "detail": f"Session '{session_id}' does not exist.",
                },
            )
        env_state = session.env.state()
        return {
            "session_id": session_id,
            "terminated": True,
            "episodes_completed": session.episodes_completed,
            "final_score": env_state.current_score,
        }

    def list_all(self) -> list[dict]:
        """Return summaries for all non-expired sessions."""
        # Prune expired sessions opportunistically
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            self._sessions.pop(sid, None)
        return [s.to_summary() for s in self._sessions.values()]

    # ── Aggregate dashboard ────────────────────────────────────────────────

    def aggregate_dashboard(self) -> dict:
        """
        Cross-session aggregate metrics for the global /dashboard endpoint.
        Only considers non-expired sessions.
        """
        active = [s for s in self._sessions.values() if not s.is_expired]
        if not active:
            return {
                "active_sessions": 0,
                "max_sessions": self.max_sessions,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "sessions": [],
            }

        gpu_utils = [s.env.ledger.gpu_utilization() for s in active]
        cpu_utils = [s.env.ledger.cpu_utilization() for s in active]
        total_arrived = sum(s.env.total_arrived for s in active)
        total_completed = sum(s.env.total_completed for s in active)

        return {
            "active_sessions": len(active),
            "max_sessions": self.max_sessions,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "aggregate": {
                "mean_gpu_utilization": round(sum(gpu_utils) / len(gpu_utils), 4),
                "mean_cpu_utilization": round(sum(cpu_utils) / len(cpu_utils), 4),
                "total_requests_arrived": total_arrived,
                "total_requests_completed": total_completed,
                "global_completion_rate": round(
                    total_completed / max(1, total_arrived), 4
                ),
            },
            "sessions": [s.to_summary() for s in active],
        }

    # ── Background cleanup ─────────────────────────────────────────────────

    async def cleanup_loop(self) -> None:
        """
        Background coroutine: sweeps expired sessions every ``cleanup_interval``
        seconds.  Start as an asyncio.Task in the FastAPI lifespan handler and
        cancel it on shutdown.
        """
        while True:
            await asyncio.sleep(self.cleanup_interval)
            expired = [
                sid for sid, s in list(self._sessions.items()) if s.is_expired
            ]
            for sid in expired:
                self._sessions.pop(sid, None)
