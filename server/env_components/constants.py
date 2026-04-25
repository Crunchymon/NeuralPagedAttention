import os

# ── Session management (overrideable via .env / environment variables) ─────────
MAX_SESSIONS        = int(os.getenv("NPA_MAX_SESSIONS", "100"))
SESSION_TTL_DEFAULT = int(os.getenv("NPA_SESSION_TTL", "3600"))   # seconds
CLEANUP_INTERVAL    = int(os.getenv("NPA_CLEANUP_INTERVAL", "60"))  # seconds
INACTIVITY_TTL      = int(os.getenv("NPA_INACTIVITY_TTL", "600"))   # 10 min default

# ── Hardware ───────────────────────────────────────────────────────────────────
GPU_TOTAL_BLOCKS = int(os.getenv("NPA_GPU_TOTAL_BLOCKS", "1000"))
CPU_TOTAL_BLOCKS = int(os.getenv("NPA_CPU_TOTAL_BLOCKS", "5000"))
TOKENS_PER_BLOCK = 16

CHATTER_RATIO = 0.80
POWER_USER_RATIO = 0.20
VIP_FREE_DEFAULT = 0.05
RETURNING_RATIO = 0.30

CHATTER_PROMPT_MU = 100
CHATTER_PROMPT_SIGMA = 50
CHATTER_GEN_MU = 250
CHATTER_GEN_SIGMA = 100

POWER_PROMPT_MU = 3500
POWER_PROMPT_SIGMA = 1200
POWER_GEN_MU = 800
POWER_GEN_SIGMA = 300

FREE_QUEUE_MAX = 100
VIP_QUEUE_MAX = 50

FREE_MULTIPLIER = 5.0
VIP_MULTIPLIER = 15.0
LATENCY_DECAY = 0.05

PAIN_95_PENALTY = -0.1
PAIN_98_PENALTY = -0.4

SWAP_TAX_FREE = -0.2
SWAP_TAX_VIP = -0.5

PREEMPT_SHRED_FREE = -1.0
PREEMPT_SHRED_VIP = -3.0
PREEMPT_SWAP_FREE = -0.6
PREEMPT_SWAP_VIP = -1.8

REJECT_UNNECESSARY_FREE = -5.0
REJECT_UNNECESSARY_VIP = -10.0
REJECT_NECESSARY_FREE = -0.5
REJECT_NECESSARY_VIP = -2.0

SLA_MISS_FREE = -10.0
SLA_MISS_VIP = -30.0
DEADLOCK_PENALTY = -20.0
CRASH_PENALTY = -100.0

INVALID_ACTION_PENALTY = -1.0
DO_NOTHING_TAX = -0.01
ADMIT_BONUS = 0.1
EVICT_BONUS = 0.05
GC_BONUS = 0.2
ACTIVE_GEN_BONUS = 0.02

GC_IDLE_THRESHOLD = 200

PHASE_CONFIGS = {
    "easy": {
        "max_ticks": 2000,
        "vip_ratio": 0.02,
        "sla_free": None,
        "sla_vip": None,
        "traffic_fn": "flat",
        "power_user_pct": 0.0,
    },
    "medium": {
        "max_ticks": 3000,
        "vip_ratio": 0.05,
        "sla_free": 100,
        "sla_vip": 50,
        "traffic_fn": "wave",
        "power_user_pct": 0.20,
    },
    "hard": {
        "max_ticks": 5000,
        "vip_ratio": 0.10,
        "sla_free": 50,
        "sla_vip": 25,
        "traffic_fn": "spike",
        "power_user_pct": 0.35,
    },
}
