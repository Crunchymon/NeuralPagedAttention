import math


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


TRAFFIC_FNS = {
    "flat": traffic_flat,
    "wave": traffic_wave,
    "spike": traffic_spike,
}