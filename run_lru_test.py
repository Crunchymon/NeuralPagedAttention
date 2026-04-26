import sys
from agents.LRUAgent.lru import run_sim

tick_logs, session_logs = run_sim("medium")
print("Medium session:", session_logs[0])
