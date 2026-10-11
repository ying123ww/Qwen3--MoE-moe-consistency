"""Record fusion capture counts after capture, without I/O in measured replay."""
import json
from pathlib import Path


def install(config):
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
    from sglang.srt.batch_invariant_ops import fused_rmsnorm_candidate as rms
    import torch.distributed as dist
    import qk_runtime_hooks as qk
    original = BreakableCudaGraphBackend.capture_one
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()

    def capture(self, shape_key, *args, **kwargs):
        before_rms = rms._GRAPH_CAPTURE_CALLS
        before_qk = qk._COUNTS['captured']
        result = original(self, shape_key, *args, **kwargs)
        graph = self._graphs[shape_key]
        row = dict(rank=rank, size=shape_key.size, mode=config['mode'],
            rms_captured_calls=rms._GRAPH_CAPTURE_CALLS-before_rms,
            qk_captured_calls=qk._COUNTS['captured']-before_qk,
            segments=len(graph._segments), breaks=len(graph._break_fns))
        (output / f'fusion-rank-{rank}-size-{shape_key.size}.json').write_text(json.dumps(row, indent=2)+'\n')
        print('FUSION_CAPTURE_PROOF '+json.dumps(row), flush=True)
        return result
    BreakableCudaGraphBackend.capture_one = capture
    return lambda module, inputs, output: None
