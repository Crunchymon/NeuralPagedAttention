from __future__ import annotations
from pydantic import BaseModel, Field


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


# Add this to the bottom of models.py
ACTION_MAP = {
    0:  "Evict Largest Free cache (idle GPU only)",
    1:  "Evict Largest VIP cache (idle GPU only)",
    2:  "Evict Oldest Free cache (idle GPU only)",
    3:  "Evict Oldest VIP cache (idle GPU only)",
    4:  "Swap Largest Free cache GPU->CPU",
    5:  "Swap Largest VIP cache GPU->CPU",
    6:  "Swap Oldest Free cache GPU->CPU",
    7:  "Swap Oldest VIP cache GPU->CPU",
    8:  "Admit next Free user from queue",
    9:  "Admit next VIP user from queue",
    10: "Reject next Free user (penalty if GPU has space)",
    11: "Reject next VIP user (penalty if GPU has space)",
    12: "Preempt & Shred Largest Active Free request",
    13: "Preempt & Shred Largest Active VIP request",
    14: "Preempt & Swap Largest Active Free -> CPU",
    15: "Preempt & Swap Largest Active VIP -> CPU",
    16: "Garbage Collect (delete idle CPU caches > 200 ticks)",
    17: "Do Nothing"
}
