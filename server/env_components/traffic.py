import math
import random

from server.env_components.request_factory import generate_request


def traffic_flat(t: int) -> int:
    """Easy phase: constant 1 request per tick."""
    return 1


def traffic_wave(t: int) -> int:
    """Medium phase: day/night wave, average ~2/tick."""
    rate = 2 + 0.005 * t + 4 * math.sin(2 * math.pi * t / 1000) + 2 * math.sin(2 * math.pi * t / 150)
    return max(0, round(rate))


def traffic_spike(t: int) -> int:
    """Hard phase: violent viral spikes."""
    rate = 4 + 0.005 * t + 8 * math.sin(2 * math.pi * t / 1000) + 4 * math.sin(2 * math.pi * t / 150)
    return max(0, round(rate))


def build_traffic_trace(
    task: str,
    max_ticks: int,
    seed: int,
    vip_ratio: float,
    power_user_pct: float,
):
    """Precompute deterministic traffic blueprints for every tick."""
    rng = random.Random(seed)
    returning_pool: list[str] = []
    trace: list[list[dict]] = []

    traffic_fn = TRAFFIC_FNS[task]
    for tick in range(max_ticks):
        n_arrivals = traffic_fn(tick) * 5
        tick_events: list[dict] = []

        for _ in range(n_arrivals):
            request_id = f"{rng.getrandbits(32):08x}"
            req = generate_request(
                tick=tick,
                vip_ratio=vip_ratio,
                power_user_pct=power_user_pct,
                returning_pool=returning_pool,
                rng=rng,
                request_id=request_id,
            )
            returning_pool.append(req.request_id)
            tick_events.append({
                "request_id": req.request_id,
                "tier": req.tier,
                "user_type": req.user_type,
                "is_returning": req.is_returning,
                "prompt_tokens": req.prompt_tokens,
                "target_gen_tokens": req.target_gen_tokens,
                "arrival_tick": tick,
            })

        trace.append(tick_events)

    return trace


TRAFFIC_FNS = {
    "flat": traffic_flat,
    "wave": traffic_wave,
    "spike": traffic_spike,
}