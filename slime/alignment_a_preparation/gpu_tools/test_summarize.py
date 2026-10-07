"""Validate that aggregation uses the slowest rank and excludes warmup."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('summary', Path(__file__).with_name('summarize.py'))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTest(unittest.TestCase):
    def test_max_rank_and_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for rank in range(8):
                (run / f'audit-rank{rank}.json').write_text(json.dumps({'variant': 'native', 'timing_scope': 'test'}))
                rows = [{'rank': rank, 'rollout_id': step, 'step_id': 0, 'elapsed_s': rank + step,
                         'grad_norm': 2, 'global_batch_size': 8, 'peak_allocated_bytes': 2**30,
                         'peak_reserved_bytes': 2**31} for step in [2, 3, 4]]
                (run / f'measurements-rank{rank}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            with patch.object(Path, 'read_bytes', return_value=b'fixed-data'):
                result = summary.summarize(run, warmup=1)
            self.assertEqual(result['step_times_max_rank_s'], [10, 11])
            self.assertEqual(result['measured_steps'], 2)
            self.assertEqual(result['step_time_median_s'], 10.5)
            self.assertEqual(result['peak_allocated_gib_max_rank'], 1)
            self.assertEqual(result['peak_reserved_gib_max_rank'], 2)
            (run / 'measurements-rank7.jsonl').write_text('')
            with self.assertRaises(AssertionError):
                summary.summarize(run, warmup=1)


if __name__ == '__main__':
    unittest.main()
