import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import memory_guard as g

class Tests(unittest.TestCase):
    def test_threshold(self):
        self.assertFalse(g.cutoff(899,900))
        self.assertTrue(g.cutoff(900,900))
        self.assertTrue(g.cutoff(901,900))

    def test_job_scope_and_self_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for pid, job, name in [(101,'job_1','python'),(102,'job_10','python'),(103,'job_1','sshd'),(104,'job_1','raylet')]:
                p=root/str(pid);p.mkdir()
                (p/'cgroup').write_text('7:memory:/slurm/'+job+'/step_extern\n')
                (p/'comm').write_text(name)
            self.assertEqual(g.targets('job_1',root,own=104),[101])

    def test_trigger_sends_sigkill(self):
        with tempfile.TemporaryDirectory() as d:
            group=Path(d)/'job_1';group.mkdir()
            (group/'memory.limit_in_bytes').write_text(str(1024*1024**3))
            (group/'memory.usage_in_bytes').write_text(str(900*1024**3))
            state=Path(d)/'state.json'
            with patch('sys.argv',['guard','--cgroup',str(group),'--state',str(state)]), patch.object(g,'targets',return_value=[123]), patch.object(g.os,'kill') as kill, patch.object(g.time,'sleep',side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):g.main()
                kill.assert_called_once_with(123,g.signal.SIGKILL)
            self.assertTrue(json.loads(state.read_text())['triggered'])
