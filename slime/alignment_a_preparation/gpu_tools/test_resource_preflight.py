import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import resource_preflight as r


class PreflightTests(unittest.TestCase):
    def healthy(self):
        return dict(cgroup_memory_readable=True, ray_cpus=32,
                    effective_memory_available_bytes=600*r.GIB,
                    temp_free_bytes=30*r.GIB, shm_free_bytes=10*r.GIB)

    def test_healthy(self):
        r.validate(self.healthy(), 512, 20)

    def test_observed_disk_shortage_blocks(self):
        state = self.healthy()
        state['temp_free_bytes'] = int(1.7*r.GIB)
        with self.assertRaisesRegex(RuntimeError, 'disk'):
            r.validate(state, 512, 20)

    def test_unknown_memory_blocks(self):
        state = self.healthy()
        state['cgroup_memory_readable'] = False
        with self.assertRaisesRegex(RuntimeError, 'cgroup'):
            r.validate(state, 512, 20)

    def test_parent_memory_and_cpu_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            child = parent/'child'
            child.mkdir()
            for directory, values in ((parent, {'memory.max': str(100*r.GIB), 'memory.current': str(90*r.GIB), 'cpu.max': '250000 100000'}),
                                      (child, {'memory.max': 'max', 'memory.current': '0', 'cpu.max': 'max 100000'})):
                for name, value in values.items():
                    (directory/name).write_text(value)
            with patch.object(r, 'cgroup_dirs', return_value=[child, parent]), patch.object(r.os, 'sched_getaffinity', return_value=set(range(32))):
                state = r.snapshot(folder)
            self.assertEqual(state['ray_cpus'], 2)
            self.assertLessEqual(state['effective_memory_available_bytes'], 10*r.GIB)
            with self.assertRaises(RuntimeError):
                r.validate(state, 512, 20)

    def test_v1_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name, value in {'memory.limit_in_bytes': str(100*r.GIB), 'memory.usage_in_bytes': str(95*r.GIB), 'cpu.cfs_quota_us':'800000', 'cpu.cfs_period_us':'100000'}.items():
                (directory/name).write_text(value)
            with patch.object(r, 'cgroup_dirs', return_value=[directory]), patch.object(r.os, 'sched_getaffinity', return_value=set(range(32))):
                state = r.snapshot(folder)
            self.assertEqual(state['ray_cpus'], 8)
            self.assertLessEqual(state['effective_memory_available_bytes'], 5*r.GIB)

if __name__ == '__main__':
    unittest.main()
