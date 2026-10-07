"""Opt-in driver wall clock for the synchronous online RL loop."""
import json
import os
from pathlib import Path
import time


class RLIterationTimer:
    def __init__(self, args, clock=time.perf_counter):
        path = os.environ.get('SLIME_RL_ITERATION_TIMING_FILE')
        self.path = Path(path) if path else None
        self.clock = clock
        if self.path:
            # This experiment intentionally measures the simple synchronous loop.
            unsupported = ('use_critic', 'release_train', 'load_debug_rollout_data',
                           'debug_train_only', 'debug_rollout_only', 'save_interval', 'eval_interval')
            if any(getattr(args, key, None) for key in unsupported):
                raise ValueError('RL iteration timing requires online actor-only loop without checkpoint/eval')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.first_rollout = None

    def start(self, rollout_id):
        if self.path:
            if self.first_rollout is None:
                self.first_rollout = rollout_id
            self.rollout_id = rollout_id
            self.started = self.previous = self.clock()
            self.phases = {}

    def mark(self, phase):
        if self.path:
            now = self.clock()
            self.phases[phase] = now - self.previous
            self.previous = now

    def finish(self):
        if self.path:
            # The last mark is after blocking onload_kv; exclude writing this record.
            row = {'rollout_id': self.rollout_id,
                   'warmup': self.rollout_id == self.first_rollout,
                   'total_s': self.previous - self.started,
                   'phases_s': self.phases}
            with self.path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print('RL_ITERATION_TIMING ' + json.dumps(row), flush=True)
