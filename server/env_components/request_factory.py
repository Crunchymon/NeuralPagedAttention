import random
import uuid

import server.env_components.constants as constants
from server.env_components.constants import (
    CHATTER_GEN_MU,
    CHATTER_GEN_SIGMA,
    CHATTER_PROMPT_MU,
    CHATTER_PROMPT_SIGMA,
    POWER_GEN_MU,
    POWER_GEN_SIGMA,
    POWER_PROMPT_MU,
    POWER_PROMPT_SIGMA,
    RETURNING_RATIO,
)
from server.env_components.state import Request


def generate_request(
    tick: int,
    vip_ratio: float,
    power_user_pct: float,
    returning_pool: list[str],
    rng: random.Random,
) -> Request:
    tier = "vip" if rng.random() < vip_ratio else "free"

    is_power = rng.random() < power_user_pct
    user_type = "power" if is_power else "chatter"

    is_returning = len(returning_pool) > 0 and rng.random() < RETURNING_RATIO

    if user_type == "chatter":
        prompt = max(16, int(rng.gauss(CHATTER_PROMPT_MU, CHATTER_PROMPT_SIGMA)))
        gen = max(16, int(rng.gauss(CHATTER_GEN_MU, CHATTER_GEN_SIGMA)))
    else:
        prompt = max(16, int(rng.gauss(POWER_PROMPT_MU, POWER_PROMPT_SIGMA)))
        gen = max(16, int(rng.gauss(POWER_GEN_MU, POWER_GEN_SIGMA)))

    max_tokens = int(constants.GPU_TOTAL_BLOCKS * 0.8 * constants.TOKENS_PER_BLOCK)
    if prompt + gen > max_tokens:
        gen = max(16, max_tokens - prompt)

    # Reusing an id from the returning pool is what lets the cache-hit path in
    # _admit_next actually fire. _admit_next defends against the rare case where
    # the same id is currently live on GPU (it rerolls a fresh uuid then).
    if is_returning:
        request_id = rng.choice(returning_pool)
    else:
        request_id = str(uuid.uuid4())[:8]

    return Request(
        request_id=request_id,
        tier=tier,
        user_type=user_type,
        is_returning=is_returning,
        prompt_tokens=prompt,
        target_gen_tokens=gen,
        arrival_tick=tick,
    )
