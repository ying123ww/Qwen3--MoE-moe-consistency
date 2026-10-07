import gzip,json
from summarize_p1 import summarize

def test_cpu_range_excludes_gpu_annotation_and_gpu_busy_is_union(tmp_path):
    p=tmp_path/'trace.json.gz'
    events=[dict(ph='X',cat='user_annotation',name='P1/train_rpc',ts=0,dur=100),
            dict(ph='X',cat='gpu_user_annotation',name='P1/train_rpc',ts=10,dur=80),
            dict(ph='X',cat='kernel',name='a',ts=10,dur=50),
            dict(ph='X',cat='kernel',name='b',ts=30,dur=50)]
    with gzip.open(p,'wt') as f:json.dump({'traceEvents':events},f)
    r=summarize(p)
    assert r['cpu_spans_s']['P1/train_rpc']==100/1e6
    assert r['gpu_sum_s']==100/1e6
    assert r['gpu_union_s']==70/1e6
