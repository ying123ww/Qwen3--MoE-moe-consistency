import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

spec=importlib.util.spec_from_file_location('p1',Path(__file__).parents[2]/'slime/utils/p1_profile.py')
p1=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)


def test_start_profile_accepts_sglang_plain_text():
    response=Mock()
    response.json.side_effect=ValueError('plain text response')
    engine=SimpleNamespace(start_profile=SimpleNamespace(remote=Mock(return_value=response)))
    manager=SimpleNamespace(get_updatable_engines_and_lock=SimpleNamespace(remote=lambda:([engine],None)))
    with patch.dict('os.environ',{'SLIME_P1_PROFILE_DIR':'/tmp/p1-test'}), patch.dict('sys.modules',{'ray':SimpleNamespace(get=lambda x:x)}):
        p1.start_rollout_profile(manager,3)
    response.raise_for_status.assert_called_once()
    response.json.assert_not_called()
