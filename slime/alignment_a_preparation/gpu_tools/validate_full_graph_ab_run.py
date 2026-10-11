"""Audit all P10 records, actual workloads, independent starts and graph execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from full_graph_ab_workload import compare


def read(path):return json.loads(path.read_text())


def audit(run):
    manifest=read(run/'manifest.json');result=read(run/'results.json')
    assert read(run/'state.json')['status']=='complete'
    assert result['independent_engines']==9 and result['formal_batches']==180
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items())
    launches=[];records={};configured={};proofs={};workload_count=0;formal_count=0
    frozen=set();commit=set();common_config=None
    for round_index in (1,2,3):
        for mode in ('A','B','C'):
            name=f'round-{round_index}-{mode}';directory=run/name
            worker_manifest=read(directory/'manifest.json');resolved=read(directory/'resolved-server-args.json')
            assert resolved['enable_deterministic_inference']==(mode=='C')
            assert resolved['cuda_graph_config']['decode']['backend']=='full'
            assert resolved['cuda_graph_config']['decode']['bs']==[2,4]
            assert resolved['cuda_graph_config']['prefill']['backend']=='disabled'
            assert resolved['prefill_max_requests'] is None and resolved['enable_return_routed_experts']
            assert worker_manifest['sampling']==dict(temperature=.6,max_new_tokens=128,ignore_eos=True)
            assert worker_manifest['environment']['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']==('1' if mode=='C' else '0')
            assert worker_manifest['environment']['SLIME_QWEN3_RESIDUAL_RMS_MODE']==('candidate' if mode=='C' else 'off')
            assert worker_manifest['environment']['SLIME_QWEN3_ROUTER_TILE_MODE']=='off'
            frozen.add(worker_manifest['frozen_sha256']);commit.add(worker_manifest['source_git_commit'])
            config={k:resolved[k] for k in ('model_path','dtype','tp_size','dp_size','attn_cp_size','ep_size',
                'moe_dp_size','moe_dense_tp_size','attention_backend','moe_runner_backend','deepep_mode','random_seed')}
            if common_config is None:common_config=config
            else:assert config==common_config
            if mode=='A':assert resolved['deepep_config'] is None
            else:
                actual=json.loads(resolved['deepep_config'])
                if configured:assert actual==configured
                else:configured=actual
            rows=[json.loads(x) for x in (directory/'measurements.jsonl').read_text().splitlines()]
            assert len(rows)==23 and [x['warmup'] for x in rows]==[True]*3+[False]*20
            assert [x['batch'] for x in rows]==list(range(23))
            formal=[x['elapsed_s'] for x in rows[3:]];formal_count+=len(formal)
            summary=read(directory/'summary.json')
            assert statistics.median(formal)==summary['median_s']
            ordered=sorted(formal);at=19*.95
            p95=ordered[int(at)]+(ordered[math.ceil(at)]-ordered[int(at)])*(at-int(at))
            assert p95==summary['p95_s']
            for row in rows:
                assert row['cached_tokens']==[0]*8 and row['output_tokens']==1024
                assert len(row['samples'])==8 and math.isfinite(row['elapsed_s']) and row['elapsed_s']>0
                with np.load(directory/row['artifact'],allow_pickle=False) as arrays:
                    for sample in row['samples']:
                        index=sample['sample'];tokens=arrays[f'tokens_{index}'];routes=arrays[f'routes_{index}']
                        assert tokens[-128:].tolist()==sample['response_ids']
                        assert routes.shape==(len(tokens)-1,48,8)
                        assert np.isfinite(arrays[f'logprob_bits_{index}'].view('<f4')).all()
                        assert hashlib.sha256(tokens.tobytes()).hexdigest()==sample['token_sha256']
                        assert hashlib.sha256(routes.tobytes()).hexdigest()==sample['route_sha256']
                workload_count+=1
            captures=[read(p) for p in (directory/'graph-evidence').glob('fixed-capture-size-*-rank-*.json')]
            replays=[read(p) for p in (directory/'graph-evidence').glob('fixed-full-replay-size-*-rank-*.json')]
            required={(r,s) for r in range(8) for s in (2,4)}
            assert {(p['rank'],p['size']) for p in captures}==required
            assert {(p['rank'],p['size']) for p in replays}==required
            assert all(p['dispatches_this_shape']==48 and p['captured_layers']==list(range(48)) for p in captures)
            runtime=[read(p) for p in (directory/'graph-evidence').glob('runtime-formal-rank-*.json')]
            assert {p['rank'] for p in runtime}==set(range(8))
            for row in runtime:
                assert row['counts'].get('full_replays',0)==2560
                assert row['counts'].get('eager_decode',0)==0 and row['counts'].get('eager_decode_idle',0)==0
                assert all(d['is_extend_in_batch'] and not d['can_run_dp_cuda_graph'] for d in row['eager_decode_details'])
                if mode!='A':assert row['communication']=={k.removeprefix('normal_'):v for k,v in configured.items()}
            first_proof_files=list((directory/'graph-evidence').glob('fixed-capture-size-*-rank-*.json'))+list(
                (directory/'graph-evidence').glob('fixed-full-replay-size-*-rank-*.json'))
            earliest_formal=(directory/'startup.json').stat().st_mtime+sum(x['elapsed_s'] for x in rows[:3])-1
            assert all(p.stat().st_mtime<earliest_formal for p in first_proof_files)
            proofs[name]=dict(full_captures=len(captures),full_first_replays=len(replays),runtime=runtime,
                evidence_before_formal=True)
            assert read(run/f'{name}-exit.json')['returncode']==0
            launches.append(read(run/f'{name}-launch.json')['pid']);records[name]=summary
    assert len(set(launches))==9 and len(commit)==len(frozen)==1
    assert commit=={result['source_git_commit']} and formal_count==180 and workload_count==207
    workload_differences={}
    for round_index in (1,2,3):
        for left,right in (('A','B'),('B','C')):
            leftdir=run/f'round-{round_index}-{left}';rightdir=run/f'round-{round_index}-{right}'
            lengths=read(leftdir/'manifest.json')['prompt_lengths']
            differences=[compare(leftdir/f'workload-{index:02d}.npz',rightdir/f'workload-{index:02d}.npz',lengths)
                         for index in range(3,23)]
            key=f'round-{round_index}-{left}-{right}'
            totals={k:sum(x[k] for x in differences) for k in differences[0]}
            assert totals==result['workload_differences'][key];workload_differences[key]=totals
    for pair in result['paired']:
        index=pair['round'];a,b,c=(records[f'round-{index}-{m}']['median_s'] for m in ('A','B','C'))
        assert pair['communication_config_overhead_percent']==100*(b/a-1)
        assert pair['deterministic_scheme_overhead_percent']==100*(c/b-1)
        assert pair['total_C_over_A_percent']==100*(c/a-1)
    return dict(passed=True,source_unchanged=True,source_git_commit=next(iter(commit)),
        independent_engines=9,unique_worker_pids=launches,formal_batches=formal_count,
        standard_workload_records=workload_count,all_uncached=True,all_full=True,
        pure_global_decode_eager_fallbacks=0,global_prefill_local_decode_events_explicitly_reported=True,
        common_resolved_config=common_config,proofs=proofs,workload_differences=workload_differences,
        scope='Observed random generate; same paths/topology/commit, actual NPZ identities checked; no online RL proof')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();result=audit(args.run)
    (args.run/'measurement-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('proofs','workload_differences')},indent=2))
