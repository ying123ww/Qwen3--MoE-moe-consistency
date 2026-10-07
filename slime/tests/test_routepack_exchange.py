"""CPU tests: synthetic loads prove constraints, not training speedup."""
from copy import deepcopy
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/routepack'))
from exchange import refine, excess
import profile_data as profile

def topology_rows(world=4):
    return [{'global_rank': r, 'world_size': world, 'vpp_size': 1,
             'groups': {'dense_dp': list(range(world)), 'ep': [r // 2 * 2, r // 2 * 2 + 1],
                        'expert_dp': list(range(r % 2, world, 2)),
                        **{name: [r] for name in ('tp', 'expert_tp', 'cp', 'pp')}},
             'experts': {'0': {'num_experts': 4, 'local_expert_indices': [r % 2 * 2, r % 2 * 2 + 1]}}}
            for r in range(world)]


def inputs():
    topology = profile.merge_topology(topology_rows())
    samples = [{'sample_id': str(i), 'step': 0, 'rollout_id': i, 'reward_group_id': i // 2,
                'model_version': 'fixture', 'expert_layout_version': 'fixture', 'token_count': 3,
                'response_length': 1, 'valid_loss_token_count': 1, 'layer_ids': [0]}
               for i in range(4)]
    traces = {str(i): np.array([[[i % 2 * 2, i % 2 * 2 + 1]]] * 2) for i in range(4)}
    schedule = {'cells': [{'step': 0, 'group_id': i // 2, 'slot': 0, 'source_rank': i, 'sample_ids': [str(i)]}
                          for i in range(4)]}
    return samples, traces, schedule, topology


NUM_GPUS = 0


def fixture():
    samples,traces,schedule,topology = inputs()
    # Two identical EP replicas; slot 0 routes to receiver 0, slot 1 to receiver 1.
    for s in samples:
        traces[s['sample_id']] = np.array([[[0,1]],[[0,1]]])
    for i in range(4):
        s = deepcopy(samples[i]);s['sample_id']=str(i+4);s['rollout_id']=i+4
        samples.append(s);traces[str(i+4)]=np.array([[[2,3]],[[2,3]]])
        c=deepcopy(schedule['cells'][i]);c['slot']=1;c['sample_ids']=[str(i+4)]
        schedule['cells'].append(c)
    return samples,traces,schedule,topology


def test_layer_stragglers_do_not_cancel():
    h=np.array([[10,0],[0,10]])
    assert excess(h)==10
    assert excess(h.sum(axis=0))==0


def test_exchange_improves_and_preserves_inputs():
    args=fixture();saved=deepcopy(args[2]);sample_saved=deepcopy(args[0])
    result,report,arrays=refine(*args,1)
    assert report['after']['E'] < report['before']['E']
    assert report['swaps']
    assert args[2]==saved and args[0]==sample_saved
    assert sorted(s for c in result['cells'] for s in c['sample_ids'])==sorted(s['sample_id'] for s in args[0])
    assert [{k:v for k,v in c.items() if k!='sample_ids'} for c in result['cells']]==[{k:v for k,v in c.items() if k!='sample_ids'} for c in saved['cells']]
    assert all(s['new_affected_E']<s['old_affected_E'] for s in report['swaps'])
    assert report['after']['padding_tokens']<=report['before']['padding_tokens']
    assert 'collective_U' in arrays
    assert refine(*args,1)[0]==result


def test_one_slot_is_noop():
    args=inputs();result,report,_=refine(*args,4)
    assert result==args[2] and report['candidates_examined']==0


@pytest.mark.parametrize('budget', [{'max_candidates':0},{'max_swaps':0}])
def test_zero_budget_preserves_native(budget):
    args=fixture();result,report,_=refine(*args,4,**budget)
    assert result==args[2] and not report['swaps']


def test_candidate_budget_and_padding_reconstruction():
    args=fixture()
    args[0][0]['token_count']=5
    args[1]['0']=np.array([[[0,1]]]*4)
    _,report,_=refine(*args,4,max_candidates=3)
    assert report['candidates_examined']<=3
    assert report['after']['padding_tokens']<=report['before']['padding_tokens']


def test_capacity_fails_closed():
    with pytest.raises(ValueError,match='capacity'):
        refine(*fixture(),4,max_bin_tokens=3)


def test_step_boundary_prevents_exchange():
    args=fixture()
    for s in args[0][4:]: s['step']=1
    for c in args[2]['cells'][4:]: c['step']=1;c['slot']=0
    result,report,_=refine(*args,1)
    assert result==args[2] and report['candidates_examined']==0


def test_duplicate_sample_rejected():
    args=fixture();args[2]['cells'][0]['sample_ids']=['1']
    with pytest.raises(ValueError,match='duplicates or drops'):
        refine(*args,1)


def test_random_same_count_and_seed():
    args=fixture();_,r,_=refine(*args,1)
    n=len(r['swaps'])
    a,ra,_=refine(*args,1,mode='random',seed=7,max_swaps=n)
    b,rb,_=refine(*args,1,mode='random',seed=7,max_swaps=n)
    assert len(ra['swaps'])==n and a==b and ra['swaps']==rb['swaps']
    assert ra['after']['padding_tokens']<=ra['before']['padding_tokens']


def test_random_accepts_non_improvement():
    args=fixture()
    for key in args[1]: args[1][key]=np.array([[[0,1]],[[0,1]]])
    _,r,_=refine(*args,1,mode='random',seed=3,max_swaps=1)
    assert len(r['swaps'])==1
    assert r['before']['E']==r['after']['E']


def test_iterations_and_wall_cutoff():
    args=fixture()
    for kwargs,reason in [({'max_iterations':0},'iteration_budget'),({'max_wall_ms':0},'wall_budget')]:
        result,r,_=refine(*args,1,**kwargs)
        assert result==args[2] and r['stop_reason']==reason
        assert r['cpu_wall_ms']['total']>=r['cpu_wall_ms']['search']>=0


def test_tie_break_is_first_native_pair():
    args=fixture();a,r,_=refine(*args,1,max_swaps=1,seed=42,candidate_batch_size=1000)
    assert r['swaps'][0]['cells']==[0,4]
    b,s,_=refine(*deepcopy(args),1,max_swaps=1,seed=42,candidate_batch_size=1000)
    assert a==b and r['swaps']==s['swaps']


def test_comparison_matches_random_count():
    from benchmark_exchange import compare
    r=compare(*fixture(),1,repeats=2,max_candidates=100,max_swaps=2,max_iterations=20)
    assert r['same_swap_count'] and r['determinism_passed']
    assert r['routepack_swaps']==r['random_swaps']


def test_sampled_best_deterministic_and_legal():
    args=fixture()
    a,r,_=refine(*args,1,mode='sampled_best',seed=4,candidate_batch_size=8,max_candidates=40,max_iterations=5)
    b,s,_=refine(*args,1,mode='sampled_best',seed=4,candidate_batch_size=8,max_candidates=40,max_iterations=5)
    assert a==b and r['swaps']==s['swaps']
    assert r['after']['E']<=r['before']['E'] and r['candidates_examined']<=40
    assert all(x['new_affected_E']<x['old_affected_E'] for x in r['swaps'])


def test_sampled_best_full_pool_selects_largest_gain():
    _,r,_=refine(*fixture(),1,mode='sampled_best',candidate_batch_size=1000,max_swaps=1)
    assert len(r['swaps'])==1
    assert r['swaps'][0]['old_affected_E']-r['swaps'][0]['new_affected_E']==4
