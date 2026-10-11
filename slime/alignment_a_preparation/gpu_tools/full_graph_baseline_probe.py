"""Attempt native full decode capture with the existing DeepEP-normal baseline.

Only this experiment bypasses ServerArgs' normal-mode graph prohibition.
There is no eager fallback; a failed capture is not a benchmark result.
"""
import json
import os
from pathlib import Path
import sys


def main():
    def value(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    run = Path(value('--run', ''))
    os.environ['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '0'
    for name in ('RESIDUAL_RMS', 'ROUTER_TILE'):
        os.environ[f'SLIME_QWEN3_{name}_MODE'] = 'off'
        os.environ[f'SLIME_QWEN3_{name}_AUDIT'] = '0'
    import sglang
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.model_executor.cuda_graph_config import Backend
    original_handle = ServerArgs._handle_a2a_moe

    def allow_full(self):
        requested = self.cuda_graph_config.decode.backend
        original_handle(self)
        if requested != Backend.FULL or self.moe_a2a_backend != 'deepep' or self.deepep_mode != 'normal' or self.enable_deterministic_inference:
            raise RuntimeError('Probe requires non-deterministic DeepEP normal full decode')
        self.cuda_graph_config.decode.backend = Backend.FULL
        (run / 'post-normal-hook.json').write_text(json.dumps(dict(
            decode_backend=str(self.cuda_graph_config.decode.backend),
            prefill_backend=str(self.cuda_graph_config.prefill.backend),
            deepep_mode=self.deepep_mode, experimental_config_bypass=True), indent=2) + '\n')
        print('FULL_GRAPH_PROBE: restored explicit full decode after normal-mode prohibition', flush=True)
    ServerArgs._handle_a2a_moe = allow_full
    original_engine = sglang.Engine

    def engine(**kwargs):
        config = dict(kwargs)
        config.update(disable_cuda_graph=False, disable_prefill_cuda_graph=True,
                      cuda_graph_config=dict(decode=dict(backend='full', bs=[4], max_bs=4),
                                             prefill=dict(backend='disabled')))
        (run / 'experimental-server-args.json').write_text(json.dumps(config, indent=2) + '\n')
        return original_engine(**config)
    sglang.Engine = engine
    import inference_benchmark
    inference_benchmark.main()


if __name__ == '__main__':
    main()
