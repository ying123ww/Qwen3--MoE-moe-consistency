"""Prepare reviewable v89-based real online arguments; does not start GPUs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import p12_experiment


def options(argv):
    result={};key=None
    for word in argv[2:]:
        if word.startswith('--'):key=word;result[key]=[]
        elif key is not None:result[key].append(word)
        else:raise ValueError('Malformed source argv')
    return result


def prepare(run,variant):
    root=Path.cwd();prep=root/'slime/alignment_a_preparation'
    source=json.loads((prep/'gpu_runs/a-restore-20260909-06/argv.json').read_text());args=options(source)
    for key in ('--sglang-enable-fp32-moe-router','--sglang-disable-cuda-graph','--sglang-trust-remote-code','--load-debug-rollout-data'):args.pop(key,None)
    args.update({'--num-rollout':['3'],'--start-rollout-id':['0'],'--lr':['1e-6'],'--save-debug-rollout-data':[str(run/'rollout/{rollout_id}.pt')],'--save-debug-train-data':[str(run/'train/{rollout_id}.pt')],'--custom-megatron-before-train-step-hook-path':['training_update_hooks.before_train_step'],'--custom-generate-function-path':['rollout_hooks.generate'],'--custom-rm-path':['reward_hooks.alternating_reward'],'--custom-rollout-log-function-path':['replay_validation_hooks.log_rollout']})
    base=p12_experiment.base.base_config();base['deepep_config']=args['--sglang-deepep-config'][0]
    config=p12_experiment.configure(base,variant,run/'graph-evidence')
    config['forward_hooks'].append(dict(name='p13-lifecycle',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='graph_lifecycle_hooks:install',config=dict(output_dir=str(run/'lifecycle'))))
    args['--sglang-cuda-graph-config']=[json.dumps(config['cuda_graph_config'])]
    args['--sglang-forward-hooks']=[json.dumps(config['forward_hooks'])]
    args['--sglang-disable-prefill-cuda-graph']=[]
    args['--router-log-level']=['info']
    argv=[source[0],source[1]]
    for key,values in args.items():argv.extend([key,*values])
    if any('$' in x for x in argv):raise RuntimeError('Unexpanded source argument')
    return dict(argv=argv,variant=variant,requested_forward_hooks=config['forward_hooks'],prerequisites=dict(p12_gate='must pass',bootstrap_env='SLIME_P13_FULL_BOOTSTRAP=1 and SLIME_P13_RUN_DIR',cuda_graph_memory_saver='0',batch_invariant_fallback_variant='0'),scope='Three real rollouts with actual StatelessAdam/reset states, two completed updates before rollout2; v89 full colocated tensor synchronization (training NCCL collection, Gloo/IPC delivery). Controlled per-group0/1 diagnosticreward,not task learning; actual GRPO gradient/Adam/sync. Not executed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--variant',choices=p12_experiment.VARIANTS,required=True);args=parser.parse_args()
    args.run=args.run.resolve();payload=prepare(args.run,args.variant);args.run.mkdir(parents=True,exist_ok=True);(args.run/'prepared-online-config.json').write_text(json.dumps(payload,indent=2)+'\n');print('P13_CONFIG_PREPARED_NOT_EXECUTED')
