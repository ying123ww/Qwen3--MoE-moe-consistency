"""Wait for asynchronous router activation before the colocated worker is offloaded."""
import functools
import json
import os
from pathlib import Path
import time


def install(module):
    original = module.SGLangEngine._register_to_router

    @functools.wraps(original)
    def register(self, server_args_dict):
        result = original(self, server_args_dict)
        if self.worker_type == 'encoder' or self.node_rank != 0 or not self.router_ip or not self.router_port:
            return result
        import requests
        session = requests.Session()
        session.trust_env = False
        base = f'http://{self.router_ip}:{self.router_port}'
        url = f'http://{self.server_host}:{self.server_port}'
        directory = Path(os.environ['SLIME_P13_RUN_DIR']) / 'router-readiness'
        directory.mkdir(exist_ok=True)
        deadline = time.monotonic() + 180
        history = []
        while True:
            row = dict(time=time.time(), worker_url=url)
            try:
                workers = session.get(base + '/workers', timeout=10)
                workers.raise_for_status()
                row['workers'] = workers.json()
                ready = session.get(base + '/readiness', timeout=10)
                row['readiness_code'] = ready.status_code
                row['readiness'] = ready.json()
                members = row['workers'].get('workers', [])
                activated = any(w.get('url') == url and w.get('is_healthy', w.get('healthy')) is True for w in members)
                row['activated'] = activated
            except Exception as exc:
                row['error'] = str(exc)
                activated = False
            history.append(row)
            (directory / 'registration.json').write_text(json.dumps(history, indent=2) + '\n')
            if activated and row.get('readiness_code') == 200:
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError('P13 worker registration accepted but router activation did not complete; see router-readiness/registration.json')
            time.sleep(1)
    module.SGLangEngine._register_to_router = register
