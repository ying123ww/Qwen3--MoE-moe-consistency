import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('optimizer_transfers', Path(__file__).with_name('analyze_optimizer_transfers.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_launch_correlation_includes_late_gpu_copy_and_rejects_unrelated_window_copy():
    events = [
        dict(cat='user_annotation', name='P1/optimizer_step', ts=10, dur=100, pid=7, tid=7),
        dict(cat='gpu_user_annotation', name='P1/optimizer_step', ts=10, dur=999, pid=0, tid=1),
        dict(cat='user_annotation', name='Optimizer.step#StatelessAdam.step', ts=20, dur=70, pid=7, tid=7),
        dict(cat='cpu_op', name='aten::copy_', ts=30, dur=30, pid=7, tid=7, args={'External id': 3}),
        dict(cat='cuda_runtime', name='cudaMemcpyAsync', ts=40, dur=10, pid=7, tid=7, args={'External id': 3, 'correlation': 12}),
        dict(cat='gpu_memcpy', name='Memcpy HtoD (Pageable -> Device)', ts=120, dur=8, pid=0, tid=1, args={'External id': 3, 'correlation': 12, 'bytes': 64}),
        dict(cat='gpu_memcpy', name='unrelated', ts=50, dur=10, pid=0, tid=1, args={'External id': 5, 'correlation': 99, 'bytes': 2048}),
    ]
    result = module.analyze(events)
    row = result['copies_by_launch_scope']['Memcpy HtoD (Pageable -> Device)']
    assert row['gpu'] == dict(count=1, duration_s=8e-6, bytes=64)
    assert row['runtime_api']['duration_s'] == 10e-6
    assert row['launches_inside_adam_scope'] == 1
    assert row['linked_cpu_operators'] == {'aten::copy_': 1}
    assert result['window_memcpy_without_optimizer_launch'] == {'unrelated': 1}
    assert result['optimizer_wall_s'] == 100e-6
