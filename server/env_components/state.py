import math
from dataclasses import dataclass
from typing import Optional

from server.env_components.constants import CPU_TOTAL_BLOCKS, GPU_TOTAL_BLOCKS, TOKENS_PER_BLOCK


@dataclass
class Request:
    request_id: str
    tier: str
    user_type: str
    is_returning: bool
    prompt_tokens: int
    target_gen_tokens: int
    generated_tokens: int = 0
    wait_ticks: int = 0
    location: str = "queue"
    blocks_allocated: int = 0
    idle_ticks: int = 0
    arrival_tick: int = 0

    @property
    def total_blocks_needed(self) -> int:
        total_tokens = self.prompt_tokens + self.target_gen_tokens
        return math.ceil(total_tokens / TOKENS_PER_BLOCK)

    @property
    def current_blocks(self) -> int:
        used_tokens = self.prompt_tokens + self.generated_tokens
        return math.ceil(used_tokens / TOKENS_PER_BLOCK)

    @property
    def is_complete(self) -> bool:
        return self.generated_tokens >= self.target_gen_tokens

    @property
    def is_active(self) -> bool:
        return self.location == "gpu_active"

    @property
    def is_idle_gpu(self) -> bool:
        return self.location == "gpu_idle"

    @property
    def is_on_cpu(self) -> bool:
        return self.location == "cpu"


class MemoryLedger:
    def __init__(self):
        self.gpu_used: int = 0
        self.cpu_used: int = 0
        self.gpu_requests: dict[str, Request] = {}
        self.cpu_requests: dict[str, Request] = {}

    def gpu_free(self) -> int:
        return GPU_TOTAL_BLOCKS - self.gpu_used

    def cpu_free(self) -> int:
        return CPU_TOTAL_BLOCKS - self.cpu_used

    def gpu_utilization(self) -> float:
        return self.gpu_used / GPU_TOTAL_BLOCKS

    def cpu_utilization(self) -> float:
        return self.cpu_used / CPU_TOTAL_BLOCKS

    def place_on_gpu(self, req: Request, location: str = "gpu_active") -> bool:
        blocks = req.current_blocks
        if blocks > self.gpu_free():
            return False
        req.location = location
        req.blocks_allocated = blocks
        self.gpu_used += blocks
        self.gpu_requests[req.request_id] = req
        return True

    def remove_from_gpu(self, req: Request):
        self.gpu_used -= req.blocks_allocated
        self.gpu_requests.pop(req.request_id, None)
        req.blocks_allocated = 0

    def place_on_cpu(self, req: Request) -> bool:
        blocks = req.current_blocks
        if blocks > self.cpu_free():
            return False
        req.location = "cpu"
        req.blocks_allocated = blocks
        self.cpu_used += blocks
        self.cpu_requests[req.request_id] = req
        return True

    def remove_from_cpu(self, req: Request):
        self.cpu_used -= req.blocks_allocated
        self.cpu_requests.pop(req.request_id, None)
        req.blocks_allocated = 0

    def idle_gpu_requests(self, tier: Optional[str] = None) -> list[Request]:
        reqs = [r for r in self.gpu_requests.values() if r.is_idle_gpu]
        if tier:
            reqs = [r for r in reqs if r.tier == tier]
        return reqs

    def active_gpu_requests(self, tier: Optional[str] = None) -> list[Request]:
        reqs = [r for r in self.gpu_requests.values() if r.is_active]
        if tier:
            reqs = [r for r in reqs if r.tier == tier]
        return reqs

    def cpu_requests_by_tier(self, tier: str) -> list[Request]:
        return [r for r in self.cpu_requests.values() if r.tier == tier]
