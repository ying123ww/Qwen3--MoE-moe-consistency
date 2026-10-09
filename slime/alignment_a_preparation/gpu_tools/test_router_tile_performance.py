"""Reject broken or incomplete P2P matrices before allocating model weights."""
import pytest
import router_tile_performance as performance


def matrix(broken=False):
    rows = ['\x1b[4mGPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7\x1b[0m']
    for source in range(8):
        cells = [('X' if source == destination else
                  'NS' if broken and 2 in (source, destination) else 'OK')
                 for destination in range(8)]
        rows.append(' GPU%d\t%s' % (source, '\t'.join(cells)))
    return '\n'.join(rows)


def test_complete_peer_matrix_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(performance.subprocess, 'check_output', lambda *a, **k: matrix())
    performance.p2p_preflight(tmp_path)
    assert (tmp_path / 'p2p-preflight.json').exists()


def test_isolated_gpu_fails_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(performance.subprocess, 'check_output', lambda *a, **k: matrix(True))
    with pytest.raises(RuntimeError, match='DeepEP requires working P2P'):
        performance.p2p_preflight(tmp_path)
    assert '"status": "NS"' in (tmp_path / 'p2p-preflight.json').read_text()


def test_incomplete_matrix_is_not_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(performance.subprocess, 'check_output', lambda *a, **k: 'GPU0 X OK')
    with pytest.raises(RuntimeError, match='Cannot establish'):
        performance.p2p_preflight(tmp_path)
