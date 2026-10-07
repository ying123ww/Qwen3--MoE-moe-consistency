"""Read only the task-owned Ray job; preserve its logs beside its outputs."""
import json
from pathlib import Path
import sys
from urllib.request import build_opener, ProxyHandler

urlopen = build_opener(ProxyHandler({})).open

job = sys.argv[1]
root = Path('/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/gpu_runs') / job
with urlopen('http://127.0.0.1:8267/api/jobs/' + job, timeout=10) as response:
    state = json.load(response)
with urlopen('http://127.0.0.1:8267/api/jobs/' + job + '/logs', timeout=10) as response:
    logs = json.load(response)['logs']
(root / 'job-status.json').write_text(json.dumps(state, indent=2) + '\n')
(root / 'job.log').write_text(logs)
print(json.dumps({k: state.get(k) for k in ['status', 'message', 'start_time', 'end_time']}))
lines = [line for line in logs.splitlines() if 'file_system_monitor.cc' not in line]
print('\n'.join(line[:1000] for line in lines[-16:]))
