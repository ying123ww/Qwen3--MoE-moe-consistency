import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from slime.utils.rl_iteration_timer import RLIterationTimer


class TimerTests(unittest.TestCase):
    def test_total_is_contiguous_and_excludes_record_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'iterations.jsonl'
            with patch.dict(os.environ, {'SLIME_RL_ITERATION_TIMING_FILE':str(path)}):
                ticks=iter([10.,13.,18.])
                timer=RLIterationTimer(SimpleNamespace(start_rollout_id=1),clock=lambda:next(ticks))
                timer.start(1); timer.mark('rollout'); timer.mark('train_sync'); timer.finish()
            row=json.loads(path.read_text())
            self.assertEqual(row['total_s'],8.)
            self.assertEqual(row['phases_s'],{'rollout':3.,'train_sync':5.})
            self.assertTrue(row['warmup'])

    def test_offline_loop_cannot_be_mislabeled_e2e(self):
        with patch.dict(os.environ, {'SLIME_RL_ITERATION_TIMING_FILE':'unused'}):
            with self.assertRaises(ValueError):
                RLIterationTimer(SimpleNamespace(debug_train_only=True))


if __name__ == '__main__':
    unittest.main()
