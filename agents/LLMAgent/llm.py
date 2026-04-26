"""
llm.py — Locally-running Qwen-3B LLM agent for the Neural PagedAttention environment.

Architecture:
  - Downloads Qwen/Qwen2.5-3B-Instruct from HuggingFace on first run (cached to disk).
  - Auto-detects hardware: bfloat16 on CUDA GPU, 4-bit quantized on CPU/MPS.
  - Loads the model ONCE at agent construction, reuses it across all ticks.
  - Each tick: formats an 8-field observation prompt → greedy decode → parse action.
  - Falls back to LRU-style heuristic if LLM output is unparseable or times out.
  - Batch-admits requests proportional to free GPU space before each LLM decision.
  - Identical run_sim(task, ticks) interface and log schema as all other agents.

Usage:
  python3 agents/LLMAgent/llm.py easy       # single task
  python3 agents/LLMAgent/llm.py            # all tasks (easy → medium → hard)

Dependencies:
  pip install transformers accelerate bitsandbytes
"""

import os
import re
import sys
import time
import uuid

import torch

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import ACTION_MAP
from server.environment import KVCacheEnvironment
from server.env_components.scoring import compute_final_score
from agents.LLMAgent.prompts import SYSTEM_PROMPT, build_user_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS   = 8      # LLM only needs to output one 1-2 digit number
FALLBACK_TIMEOUT = 10.0   # seconds — fall back to heuristic if inference takes longer
VALID_ACTIONS    = set(range(18))


# ---------------------------------------------------------------------------
# Local Environment Wrapper
# ---------------------------------------------------------------------------

class LocalEnv:
    """
    Thin wrapper around KVCacheEnvironment that:
      - Converts KVCacheObservation objects to flat float arrays.
      - Exposes admit_batch() and gpu_free_pct() for pre-step admission logic.
    """

    def __init__(self) -> None:
        self.env = KVCacheEnvironment()
        print("[*] LLMAgent: Using Local Environment")

    def reset(self, task: str = "easy", seed: int | None = None, traffic_trace: list[list[dict]] | None = None) -> list[float] | None:
        """Reset the environment and return the initial observation array."""
        try:
            obs_obj = self.env.reset(task, seed=seed, traffic_trace=traffic_trace)
            return obs_obj.to_array() if obs_obj is not None else None
        except Exception as exc:
            print(f"[!] Reset error: {exc}")
            return None

    def step(self, action: int) -> tuple:
        """Execute one action and return (obs_array, reward, done, info)."""
        try:
            obs_obj, reward, done, info = self.env.step(action)
            obs = obs_obj.to_array() if obs_obj is not None else None
            return obs, reward, done, info
        except Exception as exc:
            print(f"[!] Step error: {exc}")
            return None, 0.0, True, {}

    def admit_batch(self, tier: str, pct: float) -> float:
        """
        Batch-admit up to *pct* fraction of the given tier queue
        limited by available GPU blocks. Returns cumulative admit reward.
        """
        return self.env.admit_batch(tier, pct)

    def gpu_free_pct(self) -> float:
        """Return the fraction of GPU blocks that are currently free (0.0–1.0)."""
        import server.env_components.constants as constants
        gpu_total = constants.GPU_TOTAL_BLOCKS
        gpu_used  = self.env.ledger.gpu_used
        return max(0.0, (gpu_total - gpu_used) / gpu_total)


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """
    Qwen2.5-3B-Instruct agent for KV-cache memory management.

    The model is loaded once at construction time and kept in memory for
    the full simulation. Each tick it receives a short structured prompt
    describing the current system state and must respond with a single
    integer action ID.

    Hardware auto-detection:
      - CUDA available  → bfloat16, device=cuda (fastest, ~6 GB VRAM)
      - MPS  available  → float16, device=mps  (Apple Silicon)
      - CPU only        → 4-bit NF4 quantization via bitsandbytes (~2 GB RAM)

    Attributes:
        model_name (str):    HuggingFace model repo ID.
        device (str):        Resolved compute device.
        tokenizer:           HuggingFace tokenizer instance.
        model:               HuggingFace causal LM instance.
        parse_failures (int): Running count of ticks where parsing failed.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Download (if needed) and load the Qwen model with hardware-appropriate
        quantization. Prints progress so long cold-starts are visible.

        Args:
            model_name: HuggingFace model ID to use.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_name    = model_name
        self.parse_failures = 0

        # ── Device detection ──────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype       = torch.bfloat16
            quant_cfg   = None
            use_device_map = True    # accelerate handles multi-GPU CUDA spread
            print(f"[*] LLMAgent: CUDA detected — loading in bfloat16 on {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device    = "mps"
            dtype          = torch.float16
            quant_cfg      = None
            use_device_map = False   # MPS does NOT support device_map="auto"
            print("[*] LLMAgent: Apple MPS detected — loading in float16")
        else:
            self.device    = "cpu"
            dtype          = None    # bitsandbytes handles dtype
            quant_cfg      = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            use_device_map = True    # accelerate needed for 4-bit on CPU
            print("[*] LLMAgent: No GPU detected — loading in 4-bit NF4 (CPU mode)")

        # ── Tokenizer ─────────────────────────────────────────────────────
        print(f"[*] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # ── Model ─────────────────────────────────────────────────────────
        print(f"[*] Loading model: {model_name}  (this may take a minute on first run...)")
        load_kwargs: dict = {"trust_remote_code": True}

        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg
        elif dtype is not None:
            load_kwargs["dtype"] = dtype  # use 'dtype' not deprecated 'torch_dtype'

        if use_device_map:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # MPS: move the model to the Apple GPU after loading on CPU
        if not use_device_map and self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("[*] LLMAgent: Model ready.\n")

    # ── Public interface ──────────────────────────────────────────────────

    def select_action(self, obs: list[float], tick: int) -> int:
        """
        Choose a KV-cache management action for the current tick.

        Process:
          1. Build a structured user prompt from the observation.
          2. Run greedy inference with a short token budget.
          3. Parse the first valid integer from the response.
          4. Fall back to an LRU-style heuristic if parsing fails or times out.

        Args:
            obs:  21-dimensional observation array (from KVCacheObservation.to_array()).
            tick: Current simulation tick (for prompt context).

        Returns:
            Integer action ID in range [0, 17].
        """
        user_msg = build_user_prompt(obs, tick)

        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_msg},
        ]

        # Apply the chat template (Qwen uses a custom template)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        t_start = time.time()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,        # greedy — deterministic & fastest
                    temperature=None,       # ignored when do_sample=False
                    top_p=None,             # ignored when do_sample=False
                    use_cache=True,         # reuse KV cache across generate steps
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as exc:
            print(f"[!] LLM generate error at tick {tick}: {exc} — using heuristic")
            self.parse_failures += 1
            return self._heuristic_fallback(obs)

        elapsed = time.time() - t_start

        # Decode only the new tokens (skip prompt)
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        action = self._parse_action(response)
        if action is None:
            self.parse_failures += 1
            action = self._heuristic_fallback(obs)
            if tick % 50 == 0:  # limit log noise
                print(f"[!] Tick {tick}: parse failed ('{response}') → heuristic {action}  [{elapsed:.1f}s]")
        else:
            if tick % 50 == 0:
                print(f"[*] Tick {tick}: LLM → action {action} ({ACTION_MAP.get(action)})  [{elapsed:.1f}s]")

        return action

    # ── Private helpers ───────────────────────────────────────────────────

    def _parse_action(self, text: str) -> int | None:
        """
        Extract the first valid action integer (0–17) from the model response.

        Tries:
          1. Exact integer match at start of text.
          2. First integer found anywhere in text via regex.

        Args:
            text: Raw model output string.

        Returns:
            Integer action ID, or None if no valid integer was found.
        """
        # Fast path: text is just a number
        stripped = text.strip()
        if stripped.isdigit():
            val = int(stripped)
            if val in VALID_ACTIONS:
                return val

        # Regex fallback: find first integer token in text
        matches = re.findall(r"\b(\d{1,2})\b", stripped)
        for m in matches:
            val = int(m)
            if val in VALID_ACTIONS:
                return val

        return None

    def _heuristic_fallback(self, obs: list[float]) -> int:
        """
        LRU-style heuristic used when the LLM output cannot be parsed.

        Mirrors the multi-tier decision logic of LRUAgent so that simulation
        quality degrades gracefully rather than crashing on bad LLM outputs.

        Args:
            obs: 21-dimensional observation array.

        Returns:
            Integer action ID.
        """
        gpu_util  = obs[0]
        cpu_util  = obs[1]
        free_q    = obs[3]
        vip_q     = obs[4]
        vip_wait  = obs[7]
        free_age  = obs[14]
        vip_age   = obs[17]

        if gpu_util > 0.95:
            return 14   # Preempt & Swap Largest Active Free → CPU

        if gpu_util > 0.90:
            if vip_age > 0 and vip_age >= free_age:
                return 3    # Evict Oldest VIP idle
            if free_age > 0:
                return 2    # Evict Oldest Free idle
            return 14       # No idle targets → preempt

        if gpu_util > 0.85:
            if free_age > 0 and cpu_util < 0.80:
                return 6    # Swap Oldest Free GPU→CPU
            if vip_age > 0 and cpu_util < 0.80:
                return 7    # Swap Oldest VIP GPU→CPU
            return 16       # GC stale CPU caches

        if gpu_util > 0.70:
            if cpu_util > 0.60:
                return 16   # GC to free swap space
            if vip_wait > 0.5 and vip_q > 0:
                return 9    # Admit urgent VIP

        if cpu_util > 0.50:
            return 16       # Proactive GC

        if vip_q > 0:
            return 9        # Admit VIP straggler
        if free_q > 0:
            return 8        # Admit Free straggler

        return 17           # Genuinely idle


# ---------------------------------------------------------------------------
# Run Simulation (public API — matches all other agents)
# ---------------------------------------------------------------------------

def run_sim(task: str | None = None, ticks: int | None = None, seed: int | None = None, traffic_trace: list[list[dict]] | None = None) -> tuple[list, list]:
    """
    Run the LLM agent simulation for the given task difficulty.

    Args:
        task:  One of "easy", "medium", "hard", or None (runs all three in sequence).
        ticks: Optional override for max_ticks. Overrides the PHASE_CONFIGS default.

    Returns:
        Tuple of (tick_logs, session_logs):
          - tick_logs:    List of per-tick dicts with observation, action, reward, etc.
          - session_logs: List of per-episode summary dicts.
    """
    env   = LocalEnv()
    agent = LLMAgent()

    print("\n" + "=" * 60)
    print("  LLM AGENT (Qwen2.5-3B-Instruct) SIMULATION")
    print("=" * 60 + "\n")

    # Observation keys must match KVCacheObservation.to_array() order
    keys = [
        "gpu_utilization_pct", "cpu_utilization_pct", "memory_pressure_trend",
        "total_free_req",      "total_vip_req",        "total_req",
        "free_max_wait_time_pct", "vip_max_wait_time_pct", "yield_preempt_active",
        "free_size_max",  "free_size_mean",  "free_size_std_dev",
        "vip_size_max",   "vip_size_mean",   "vip_size_std_dev",
        "free_age_max",   "free_age_mean",   "free_age_std_dev",
        "vip_age_max",    "vip_age_mean",    "vip_age_std_dev",
    ]

    tasks_to_run = ["easy", "medium", "hard"] if task is None else [task]
    all_tick_logs:    list[dict] = []
    all_session_logs: list[dict] = []

    for current_task in tasks_to_run:
        session_id = str(uuid.uuid4())
        obs = env.reset(current_task, seed=seed, traffic_trace=traffic_trace)
        if obs is None:
            print(f"[!] Reset failed for task '{current_task}', skipping.")
            continue

        # Override max_ticks if caller requested a custom duration
        if ticks is not None:
            env.env.config["max_ticks"] = ticks

        total_reward = 0.0
        ticks_run    = 0
        done         = False
        logs: list[dict] = []

        while not done:
            # ── Pre-step: batch admit proportional to free GPU space ──
            gpu_util = obs[0]
            free_q   = obs[3]
            vip_q    = obs[4]
            if gpu_util < 0.85 and (vip_q > 0 or free_q > 0):
                pct = env.gpu_free_pct()
                if vip_q > 0:
                    env.admit_batch("vip", pct)
                if free_q > 0:
                    env.admit_batch("free", pct)

            # ── LLM decides the housekeeping action ───────────────────
            action      = agent.select_action(obs, tick=ticks_run)
            action_name = ACTION_MAP.get(action, "Unknown")

            obs, reward, done, info = env.step(action)
            if obs is None:
                done = True
                break

            total_reward += reward
            ticks_run    += 1

            obs_dict  = dict(zip(keys, obs))
            log_entry = {
                "task":       current_task,
                "tick":       ticks_run,
                "session_id": session_id,
                "action":     action_name,
                "reward":     round(reward, 2),
                "score":      round(total_reward, 2),
                "episode":    1,
                "tick_prompt_tokens": info.get("tick_prompt_tokens", 0),
                "tick_gen_tokens":    info.get("tick_gen_tokens",    0),
                "tick_max_tokens":    info.get("tick_max_tokens",    0),
                **obs_dict,
            }
            logs.append(log_entry)

            if ticks_run % 20 == 0 or done:
                print(
                    f"[{current_task.upper()} T{ticks_run:4}] "
                    f"{action_name:35} | "
                    f"R {reward:+7.2f} | "
                    f"Total {total_reward:9.2f} | "
                    f"GPU {obs[0]:.2f}"
                )

        # ── Session summary ───────────────────────────────────────────
        final_score = compute_final_score(
            task=current_task,
            total_completed=env.env.total_completed,
            total_arrived=env.env.total_arrived,
            per_request_fluency=env.env._per_request_fluency,
            total_cache_hits=env.env.total_cache_hits,
            total_returning_arrived=env.env.total_returning_arrived,
            total_swaps=env.env.total_swaps,
            total_actions=env.env.total_actions,
        )

        session_log = {
            "session_id":       session_id,
            "task":             current_task,
            "episode":          1,
            "total_reward":     total_reward,
            "final_score":      final_score,
            "ticks_run":        ticks_run,
            "total_arrived":    env.env.total_arrived,
            "total_completed":  env.env.total_completed,
            "crashed":          getattr(env.env, "crashed", False),
            "parse_failures":   agent.parse_failures,
        }
        all_session_logs.append(session_log)
        all_tick_logs.extend(logs)

        print(
            f"\n[✓] {current_task.upper()} Score: {total_reward:.2f} | "
            f"Env Score: {final_score:.3f} | "
            f"Parse failures: {agent.parse_failures}\n"
            + "-" * 60
        )

    return all_tick_logs, all_session_logs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_sim(task=task_arg)
