#!/usr/bin/env python3
"""
Hugging Face Space entrypoint: PPO multi-GPU DDP training.

Boots a Gradio UI, runs `accelerate launch` over agents/UnslothPPOAgent/train_ppo_ddp.py
across all GPUs visible to the Space (designed for 4x Nvidia L4 hardware), tails the
log, and optionally uploads the trained LoRA adapter to a Hub model repo.

Repository SECRETS (Settings → Secrets):
  HF_TOKEN          required for uploading; also lets gated models (Qwen) download faster

Repository VARIABLES (Settings → Variables):
  AUTO_TRAIN        1 = auto-start when the Space boots (default 1). Set 0 to start manually.
  PPO_MODEL_NAME    HF model id. Default: Qwen/Qwen2.5-7B-Instruct
                    Other tested options:
                      Qwen/Qwen2.5-3B-Instruct (fast iteration)
                      Qwen/Qwen2.5-14B-Instruct (tight upper bound on 24 GB L4)
  PPO_EPISODES      total episodes (default 80)
  PPO_LR            learning rate (default 2e-5)
  PPO_ENTROPY       entropy coefficient (default 0.04)
  PPO_MAX_TICKS     cap ticks per episode (empty = use task default)
  PPO_CURRICULUM    episodes pinned to 'easy' at start (-1 = auto = 25%% of total)
  PPO_SAVE_EVERY    save checkpoint every N episodes (0 = end only)
  PPO_NO_4BIT       1 to disable BNB 4-bit and use bf16 LoRA only (memory fallback)
  PPO_SEED          random seed (default 42)
  OUTPUT_REPO_ID    e.g. your-user/npa-ppo — create an empty model repo first
  ACCEL_CONFIG      override accelerate config path (default hf_ppo_space/accelerate_config.yaml)
  LOG_UPLOAD_INTERVAL_SEC  seconds between in-flight log uploads (default 60)

Each run gets a unique RUN_ID (UTC timestamp). When OUTPUT_REPO_ID is set, the
training stdout (`train.log`) and per-episode metrics (`metrics.jsonl`) are pushed
to `OUTPUT_REPO_ID/logs/<RUN_ID>/` every LOG_UPLOAD_INTERVAL_SEC seconds while
training is running, plus one guaranteed final upload at the end (success or fail),
so the full training log survives the Space being paused or restarted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_PATH = Path("/tmp/npa_ppo.log")
METRICS_PATH = Path("/tmp/npa_ppo_metrics.jsonl")
ACCEL_CFG_DEFAULT = str(ROOT / "hf_ppo_space" / "accelerate_config.yaml")
TRAINER = str(ROOT / "agents" / "UnslothPPOAgent" / "train_ppo_ddp.py")
ADAPTER_DIR = ROOT / "agents" / "UnslothPPOAgent" / "ppo_lora_agent"

LOG_UPLOAD_INTERVAL_SEC = int(os.environ.get("LOG_UPLOAD_INTERVAL_SEC", "60"))

STATE: dict = {"status": "idle", "error": None, "run_id": None}
_TRAIN_LOCK = threading.Lock()
_TRAIN_RUNNING = False


def _append_log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


def _build_cmd() -> list[str]:
    cfg = os.environ.get("ACCEL_CONFIG", "").strip() or ACCEL_CFG_DEFAULT

    cmd = [
        "accelerate", "launch",
        "--config_file", cfg,
        TRAINER,
        "--model-name", os.environ.get("PPO_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        "--episodes", os.environ.get("PPO_EPISODES", "80"),
        "--lr", os.environ.get("PPO_LR", "2e-5"),
        "--entropy-coef", os.environ.get("PPO_ENTROPY", "0.04"),
        "--seed", os.environ.get("PPO_SEED", "42"),
        "--save-every", os.environ.get("PPO_SAVE_EVERY", "0"),
        "--metrics-jsonl", str(METRICS_PATH),
    ]

    mt = os.environ.get("PPO_MAX_TICKS", "").strip()
    if mt:
        cmd.extend(["--max-ticks", mt])

    curric = os.environ.get("PPO_CURRICULUM", "").strip()
    if curric:
        cmd.extend(["--curriculum-episodes", curric])

    if os.environ.get("PPO_NO_4BIT", "").lower() in ("1", "true", "yes"):
        cmd.append("--no-4bit")

    return cmd


def _upload_log_snapshot(repo_id: str, token: str, run_id: str, tag: str = "snapshot") -> None:
    """Push the current /tmp log + metrics jsonl to OUTPUT_REPO_ID/logs/<run_id>/."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    base = f"logs/{run_id}"
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
            api.upload_file(
                path_or_fileobj=str(LOG_PATH),
                path_in_repo=f"{base}/train.log",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"PPO training log {tag} ({run_id})",
            )
        if METRICS_PATH.exists() and METRICS_PATH.stat().st_size > 0:
            api.upload_file(
                path_or_fileobj=str(METRICS_PATH),
                path_in_repo=f"{base}/metrics.jsonl",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"PPO metrics {tag} ({run_id})",
            )
    except Exception as exc:
        print(f"[!] log snapshot upload failed ({tag}): {exc}", flush=True)


def _periodic_log_uploader(repo_id: str, token: str, run_id: str,
                           interval: int, stop_event: threading.Event) -> None:
    """While training is running, push the log + metrics every `interval` seconds."""
    while not stop_event.wait(interval):
        _upload_log_snapshot(repo_id, token, run_id, tag="snapshot")


def _stream_subprocess(cmd: list[str]) -> int:
    _append_log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_log(line.rstrip("\n"))
    proc.wait()
    return proc.returncode


def _run_training() -> None:
    global _TRAIN_RUNNING
    with _TRAIN_LOCK:
        if _TRAIN_RUNNING:
            return
        _TRAIN_RUNNING = True

    run_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    STATE["status"] = "running"
    STATE["error"] = None
    STATE["run_id"] = run_id
    LOG_PATH.write_text("", encoding="utf-8")
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()

    log_thread: threading.Thread | None = None
    log_stop_event = threading.Event()
    repo_id = (os.environ.get("OUTPUT_REPO_ID") or "").strip()
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()

    try:
        if tok:
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

        _append_log(f"Run ID: {run_id}")
        _append_log(f"Working directory: {ROOT}")
        _append_log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        try:
            _append_log("nvidia-smi:")
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
            _append_log(r.stdout or r.stderr or "(no nvidia-smi output)")
        except Exception as exc:
            _append_log(f"(nvidia-smi unavailable: {exc})")

        if repo_id and tok:
            _append_log(f"[*] Periodic log uploader: every {LOG_UPLOAD_INTERVAL_SEC}s "
                        f"→ {repo_id}/logs/{run_id}/")
            log_thread = threading.Thread(
                target=_periodic_log_uploader,
                args=(repo_id, tok, run_id, LOG_UPLOAD_INTERVAL_SEC, log_stop_event),
                daemon=True,
            )
            log_thread.start()
        else:
            _append_log("[*] OUTPUT_REPO_ID or HF_TOKEN missing — periodic log upload disabled.")

        _append_log("========== PPO DDP ==========")
        rc = _stream_subprocess(_build_cmd())
        if rc != 0:
            STATE["error"] = f"PPO trainer failed (exit {rc})"
            STATE["status"] = "failed"
            # fall through to finally so logs still get uploaded
            return

        if repo_id and tok:
            _append_log(f"========== Upload adapter to {repo_id} ==========")
            try:
                _upload_artifacts(repo_id, tok)
                _append_log("Adapter upload finished.")
            except Exception as exc:
                STATE["error"] = f"Upload error: {exc}"
                _append_log(str(exc))
                STATE["status"] = "failed"
                return
        elif repo_id and not tok:
            _append_log("OUTPUT_REPO_ID set but HF_TOKEN missing — skipping adapter upload.")
        else:
            _append_log("No OUTPUT_REPO_ID — skipping upload. Download from Space Files tab "
                        "or duplicate the Space disk to keep the adapter.")

        STATE["status"] = "done"
    except Exception as exc:
        STATE["error"] = str(exc)
        STATE["status"] = "failed"
        _append_log(str(exc))
    finally:
        # Stop the periodic uploader and do one guaranteed final flush so the full,
        # post-training log lands in the Hub repo regardless of success/failure.
        log_stop_event.set()
        if log_thread is not None:
            log_thread.join(timeout=10)
        if repo_id and tok:
            try:
                _append_log(f"[*] Final log upload → {repo_id}/logs/{run_id}/")
                _upload_log_snapshot(repo_id, tok, run_id, tag="final")
            except Exception as exc:
                _append_log(f"[!] Final log upload failed: {exc}")
        _TRAIN_RUNNING = False


def _upload_artifacts(repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi, login

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    if ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.iterdir()):
        api.upload_folder(
            folder_path=str(ADAPTER_DIR),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="ppo_lora_agent",
            commit_message="Add PPO LoRA adapter (DDP run)",
        )
        _append_log(f"Uploaded {ADAPTER_DIR.name}/ to {repo_id}/ppo_lora_agent/")
    else:
        _append_log(f"Adapter directory {ADAPTER_DIR} is empty — nothing to upload.")


def _poll() -> tuple[str, str]:
    text = ""
    if LOG_PATH.exists():
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        if len(text) > 16000:
            text = text[-16000:]
    status = f"Status: {STATE['status']}"
    if STATE.get("run_id"):
        status += f" | run_id={STATE['run_id']}"
    if STATE.get("error"):
        status += f" | {STATE['error']}"
    return text, status


def _start_manual() -> str:
    if _TRAIN_RUNNING:
        return "Already running."
    threading.Thread(target=_run_training, daemon=True).start()
    time.sleep(0.5)
    return "Started training thread."


with gr.Blocks(title="NeuralPagedAttention — PPO 4xL4") as demo:
    gr.Markdown(
        "## NeuralPagedAttention — PPO multi-GPU training (Hugging Face Space)\n"
        "This Space launches `accelerate launch` over `agents/UnslothPPOAgent/train_ppo_ddp.py` "
        "across all visible GPUs. Configure the run via Space **Variables** "
        "(`PPO_MODEL_NAME`, `PPO_EPISODES`, …) and **Secrets** (`HF_TOKEN`, `OUTPUT_REPO_ID`). "
        "Default base model: `Qwen/Qwen2.5-7B-Instruct`."
    )
    status_box = gr.Textbox(label="Status", interactive=False)
    log_box = gr.Textbox(label="Log (tail)", lines=30, interactive=False, max_lines=40)
    start_btn = gr.Button("Start training (only if not already running)")
    start_btn.click(_start_manual, outputs=status_box)
    demo.load(_poll, outputs=[log_box, status_box], every=4)


if __name__ == "__main__":
    if os.environ.get("AUTO_TRAIN", "1").strip().lower() not in ("0", "false", "no"):
        threading.Thread(target=_run_training, daemon=True).start()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
