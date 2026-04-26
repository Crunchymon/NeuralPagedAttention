import sys
from agents.QLearningAgent.QAgent import run_sim

tick_logs, session_logs = run_sim()
for log in session_logs:
    print(f"Task: {log['task']}, Crashed: {log['crashed']}, Ticks: {log['ticks_run']}")

# Let's print the first 10 ticks of medium
medium_logs = [l for l in tick_logs if l['task'] == 'medium']
for i in range(min(15, len(medium_logs))):
    print(f"Tick: {medium_logs[i]['tick']}, GPU: {medium_logs[i]['gpu_utilization_pct']:.2f}")

