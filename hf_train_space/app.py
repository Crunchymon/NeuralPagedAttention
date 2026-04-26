#!/usr/bin/env python3
"""
Hugging Face Space entrypoint: run DQN (and optionally PPO) training, stream logs,
optionally upload checkpoints to a Hub model repo.

Secrets / variables (Space Settings → Repository secrets):
  HF_TOKEN          — required for upload; also use for gated models (Qwen)
  OUTPUT_REPO_ID    — e.g. your-user/npa-checkpoints (create empty model repo first)

Variables (Space Settings → Variables):
  AUTO_TRAIN        — 1 = start training when Space boots (default 1)
  DQN_EPISODES      — default 500
  DQN_EVAL_EVERY    — default 50
  SEED              — default 42
  TRAIN_PPO         — 1 to run agents/UnslothPPOAgent/train_ppo.py after DQN (needs Unsloth image)
  PPO_EPISODES      — default 80
  PPO_MAX_TICKS     — optional cap, e.g. 1200 (empty = full task horizon)
  PPO_LR            — default 2e-5
  PPO_ENTROPY       — default 0.05
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

LOG_PATH = Path("/tmp/npa_train.log")
STATE: dict = {"status": "idle", "error": None}
_TRAIN_LOCK = threading.Lock()
_TRAIN_RUNNING = False


def _append_log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


def _run_training() -> None:
    global _TRAIN_RUNNING
    with _TRAIN_LOCK:
        if _TRAIN_RUNNING:
            return
        _TRAIN_RUNNING = True

    STATE["status"] = "running"
    STATE["error"] = None
    LOG_PATH.write_text("", encoding="utf-8")

    try:
        tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

        _append_log(f"Working directory: {ROOT}")
        _append_log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

        # --- DQN ---
        _append_log("========== DQN ==========")
        dqn_cmd = [
            sys.executable,
            str(ROOT / "train_dqn.py"),
            "--episodes",
            os.environ.get("DQN_EPISODES", "500"),
            "--eval-every",
            os.environ.get("DQN_EVAL_EVERY", "50"),
            "--seed",
            os.environ.get("SEED", "42"),
        ]
        if os.environ.get("DQN_FAST", "").lower() in ("1", "true", "yes"):
            dqn_cmd.append("--fast")

        r = subprocess.run(dqn_cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _append_log(r.stdout or "")
        if r.returncode != 0:
            STATE["error"] = f"DQN failed (exit {r.returncode})"
            STATE["status"] = "failed"
            return

        # --- Optional PPO ---
        if os.environ.get("TRAIN_PPO", "").lower() in ("1", "true", "yes"):
            _append_log("========== PPO / Unsloth ==========")
            ppo = [
                sys.executable,
                str(ROOT / "agents" / "UnslothPPOAgent" / "train_ppo.py"),
                "--episodes",
                os.environ.get("PPO_EPISODES", "80"),
                "--lr",
                os.environ.get("PPO_LR", "2e-5"),
                "--entropy-coef",
                os.environ.get("PPO_ENTROPY", "0.05"),
            ]
            mt = os.environ.get("PPO_MAX_TICKS", "").strip()
            if mt:
                ppo.extend(["--max-ticks", mt])
            r2 = subprocess.run(ppo, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            _append_log(r2.stdout or "")
            if r2.returncode != 0:
                STATE["error"] = f"PPO failed (exit {r2.returncode})"
                STATE["status"] = "failed"
                return

        # --- Upload ---
        repo_id = (os.environ.get("OUTPUT_REPO_ID") or "").strip()
        if repo_id and tok:
            _append_log(f"========== Upload to {repo_id} ==========")
            try:
                _upload_artifacts(repo_id, tok)
                _append_log("Upload finished.")
            except Exception as exc:
                STATE["error"] = f"Upload error: {exc}"
                _append_log(str(exc))
                STATE["status"] = "failed"
                return
        elif repo_id and not tok:
            _append_log("OUTPUT_REPO_ID set but HF_TOKEN missing — skip upload.")
        else:
            _append_log("No OUTPUT_REPO_ID — skip upload. Download weights from Space Files or duplicate Space disk.")

        STATE["status"] = "done"
    except Exception as exc:
        STATE["error"] = str(exc)
        STATE["status"] = "failed"
        _append_log(str(exc))
    finally:
        _TRAIN_RUNNING = False


def _upload_artifacts(repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi, login

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    pairs = [
        (ROOT / "agents" / "NeuralAgent" / "dqn_weights_best.pth", "dqn_weights_best.pth"),
        (ROOT / "agents" / "NeuralAgent" / "dqn_weights.pth", "dqn_weights.pth"),
    ]
    for path, name in pairs:
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {name}",
            )
            _append_log(f"Uploaded {name}")

    adapter = ROOT / "agents" / "UnslothPPOAgent" / "ppo_lora_agent"
    if adapter.is_dir() and any(adapter.iterdir()):
        api.upload_folder(
            folder_path=str(adapter),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="ppo_lora_agent",
            commit_message="Add PPO LoRA adapter",
        )
        _append_log("Uploaded ppo_lora_agent/")


def _poll() -> tuple[str, str]:
    text = ""
    if LOG_PATH.exists():
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        if len(text) > 16000:
            text = text[-16000:]
    status = f"Status: {STATE['status']}"
    if STATE.get("error"):
        status += f" | {STATE['error']}"
    return text, status


def _start_manual() -> str:
    if _TRAIN_RUNNING:
        return "Already running."
    threading.Thread(target=_run_training, daemon=True).start()
    time.sleep(0.5)
    return "Started training thread."


with gr.Blocks(title="NeuralPagedAttention training") as demo:
    gr.Markdown(
        "## NeuralPagedAttention — GPU training on Hugging Face\n"
        "Training runs on this Space’s GPU. Set **Secrets**: `HF_TOKEN`, optional `OUTPUT_REPO_ID` "
        "(model repo). Toggle **Variables** for episode counts. Logs refresh every few seconds."
    )
    status_box = gr.Textbox(label="Status", interactive=False)
    log_box = gr.Textbox(label="Log (tail)", lines=28, interactive=False, max_lines=40)
    start_btn = gr.Button("Start training (only if not already running)")
    start_btn.click(_start_manual, outputs=status_box)
    demo.load(_poll, outputs=[log_box, status_box], every=4)

if __name__ == "__main__":
    if os.environ.get("AUTO_TRAIN", "1").strip().lower() not in ("0", "false", "no"):
        threading.Thread(target=_run_training, daemon=True).start()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
