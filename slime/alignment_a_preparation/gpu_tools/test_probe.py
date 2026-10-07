"""CPU-only checks for the opt-in timing wrapper's boundaries and contract."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('alignment_a_probe.py'))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.args = SimpleNamespace(qwen3_alignment_layers=None, lr=0)
        cuda = SimpleNamespace(
            synchronize=lambda: self.events.append('synchronize'),
            reset_peak_memory_stats=lambda: self.events.append('reset'),
            memory_allocated=lambda: 10, max_memory_allocated=lambda: 20,
            max_memory_reserved=lambda: 30, get_device_name=lambda: 'fake')
        dist = SimpleNamespace(get_rank=lambda: 0, barrier=lambda: self.events.append('barrier'))
        torch = ModuleType('torch'); torch.Tensor = type('Tensor', (), {})
        torch.cuda = cuda; torch.distributed = dist; torch.__version__ = 'fake'
        self.training = ModuleType('training')
        self.training.train_one_step = lambda *a, **kw: (self.events.append('original') or ({'loss': 1}, 2.0))
        pkg = ModuleType('slime.backends.megatron_utils'); pkg.model = self.training
        core = ModuleType('megatron.core')
        core.parallel_state = SimpleNamespace(get_data_parallel_world_size=lambda: 8,
             get_expert_model_parallel_world_size=lambda: 4, get_expert_model_parallel_rank=lambda: 0)
        deep = ModuleType('deep_ep')
        deep.Buffer = SimpleNamespace(num_sms=20, get_dispatch_config=self.selector, get_combine_config=self.selector)
        self.modules = {'torch': torch, 'torch.distributed': dist,
                        'slime.backends.megatron_utils': pkg, 'megatron.core': core, 'deep_ep': deep}
        self.model = [SimpleNamespace(modules=lambda: [])]

    @staticmethod
    def selector(size):
        return size

    def test_wrapper_keeps_result_and_records_complete_step(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, self.modules), patch.dict(os.environ, {'ALIGN_A_VARIANT': 'native', 'ALIGN_A_RUN_DIR': tmp}):
            probe.before_train_step(self.args, 1, 0, self.model, None, None)
            wrapped = self.training.train_one_step
            probe.before_train_step(self.args, 1, 0, self.model, None, None)
            self.assertIs(wrapped, self.training.train_one_step)
            result = wrapped(self.args, 2, 0, [], self.model, None, None, 1, 8)
            self.assertEqual(result, ({'loss': 1}, 2.0))
            self.assertEqual(self.events, ['barrier', 'synchronize', 'reset', 'original', 'synchronize'])
            row = json.loads((Path(tmp) / 'measurements-rank0.jsonl').read_text())
            self.assertEqual((row['rollout_id'], row['num_microbatches'], row['global_batch_size']), (2, 1, 8))
            self.assertEqual((row['peak_allocated_bytes'], row['peak_reserved_bytes']), (20, 30))
            self.assertGreaterEqual(row['elapsed_s'], 0)

    def test_native_rejects_non_none_layer_selection(self):
        self.args.qwen3_alignment_layers = []
        with patch.dict(sys.modules, self.modules), patch.dict(os.environ, {'ALIGN_A_VARIANT': 'native'}):
            with self.assertRaisesRegex(RuntimeError, 'alignment state'):
                probe.before_train_step(self.args, 1, 0, self.model, None, None)


if __name__ == '__main__':
    unittest.main()
