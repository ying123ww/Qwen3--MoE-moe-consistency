"""User-requested direct RoutePack timing; new 128-sample gate skipped."""
import json
import os
import socket
import autopack_gpu_compare128_sequence as case


def main():
    if socket.gethostname().split('.')[0] != 'ai_hgx_05':
        raise RuntimeError('ai_hgx_05 only')
    sequence=case.sequence
    sequence.command_for=case.command_for
    os.environ['ROUTEPACK_CHECKPOINT_CPU_OFFLOAD']='0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    record=dict(random='cancelled_by_user',new_128_gate='skipped_by_explicit_user_request',
                prior_256_gate='passed; does not replace the skipped 128-sample gate')
    path=sequence.BASE/'routepack-only-status.json'
    path.write_text(json.dumps(dict(record,status='starting'),indent=2)+'\n')
    try:
        result=sequence.run_stage('timing','routepack')
    except Exception as exc:
        path.write_text(json.dumps(dict(record,status='failed',error=str(exc)),indent=2)+'\n')
        raise
    path.write_text(json.dumps(dict(record,status='passed',timing=result),indent=2)+'\n')


if __name__=='__main__':
    main()
