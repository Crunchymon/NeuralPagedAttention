from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class KVCacheObservation(BaseModel):
    """20-dimensional normalized observation vector."""
    # Macro Health
    gpu_utilization_pct: float = Field(ge=0.0, le=1.0)
    cpu_utilization_pct: float = Field(ge=0.0, le=1.0)
    memory_pressure_trend: float = Field(ge=-1.0, le=1.0)
    # Queue Status
    free_queue_pressure: float = Field(ge=0.0, le=1.0)
    vip_queue_pressure: float = Field(ge=0.0, le=1.0)
    free_max_wait_time_pct: float = Field(ge=0.0, le=1.0)
    vip_max_wait_time_pct: float = Field(ge=0.0, le=1.0)
    # Action Yields
    yield_preempt_active: float = Field(ge=0.0, le=1.0)
    free_size_max: float = Field(ge=0.0, le=1.0)
    free_size_mean: float = Field(ge=0.0, le=1.0)
    free_size_std_dev: float = Field(ge=0.0, le=1.0)
    vip_size_max: float = Field(ge=0.0, le=1.0)
    vip_size_mean: float = Field(ge=0.0, le=1.0)
    vip_size_std_dev: float = Field(ge=0.0, le=1.0)
    free_age_max: float = Field(ge=0.0, le=1.0)
    free_age_mean: float = Field(ge=0.0, le=1.0)
    free_age_std_dev: float = Field(ge=0.0, le=1.0)
    vip_age_max: float = Field(ge=0.0, le=1.0)
    vip_age_mean: float = Field(ge=0.0, le=1.0)
    vip_age_std_dev: float = Field(ge=0.0, le=1.0)

    def to_array(self) -> list[float]:
        """Return observation as a flat list in index order 1-20."""
        return [
            self.gpu_utilization_pct, self.cpu_utilization_pct,
            self.memory_pressure_trend, self.free_queue_pressure,
            self.vip_queue_pressure, self.free_max_wait_time_pct,
            self.vip_max_wait_time_pct, self.yield_preempt_active,
            self.free_size_max, self.free_size_mean, self.free_size_std_dev,
            self.vip_size_max, self.vip_size_mean, self.vip_size_std_dev,
            self.free_age_max, self.free_age_mean, self.free_age_std_dev,
            self.vip_age_max, self.vip_age_mean, self.vip_age_std_dev,
        ]


class KVCacheAction(BaseModel):
    """Single discrete action, integer 0-17."""
    action_id: int = Field(ge=0, le=17)


class KVCacheState(BaseModel):
    """Episode metadata returned by state()."""
    episode_id: str
    task: str                      # "easy" | "medium" | "hard"
    tick: int
    max_ticks: int
    total_arrived: int
    total_completed: int
    total_rejected: int
    total_crashed: bool
    current_score: float
    cumulative_reward: float


class StepResult(BaseModel):
    observation: KVCacheObservation
    reward: float
    done: bool
    info: dict
