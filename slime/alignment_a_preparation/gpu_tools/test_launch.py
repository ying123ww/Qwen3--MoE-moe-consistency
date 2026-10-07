import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import launch
from resource_preflight import GIB


class LaunchTests(unittest.TestCase):
    def test_low_disk_aborts_before_gpu_or_cluster_commands(self):
        state = dict(cgroup_memory_readable=True, ray_cpus=32,
                     effective_memory_available_bytes=600*GIB,
                     temp_free_bytes=int(1.7*GIB), shm_free_bytes=10*GIB)
        with tempfile.TemporaryDirectory() as folder:
            prep = Path(folder)
            (prep/'gpu_runs').mkdir()
            with patch.object(launch, 'PREP', prep), patch.object(launch.socket, 'gethostname', return_value='ai_hgx_03'), patch.object(launch, 'snapshot', return_value=state), patch.object(launch.subprocess, 'check_output') as query, patch.object(launch.subprocess, 'run') as start, patch('sys.argv', ['launch.py', '--mode', 'restore', '--run-id', 'test']):
                with self.assertRaisesRegex(RuntimeError, 'disk'):
                    launch.main()
                query.assert_not_called()
                start.assert_not_called()
                self.assertTrue((prep/'gpu_runs/test/resource-preflight.json').exists())

if __name__ == '__main__':
    unittest.main()
