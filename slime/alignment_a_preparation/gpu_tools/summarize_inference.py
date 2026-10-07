"""Validate every inference batch before producing paired descriptive results."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'gpu_runs'


def main():
    pairs = []
    configs = {}
    for pair in range(1,4):
        summaries = {}
        for variant in ['baseline','aligned']:
            run = RUNS/f'a-inference-{variant}-20260910-{pair:02d}'
            assert json.loads((run/'exit.json').read_text())['returncode'] == 0
            rows = [json.loads(line) for line in (run/'measurements.jsonl').read_text().splitlines()]
            assert [r['batch'] for r in rows] == list(range(23))
            assert all(r['warmup'] == (r['batch']<3) and r['output_tokens']==1024 and r['elapsed_s']>0 for r in rows)
            summary = json.loads((run/'summary.json').read_text())
            assert summary['median_s'] == statistics.median(r['elapsed_s'] for r in rows[3:])
            manifest = json.loads((run/'manifest.json').read_text())
            assert manifest['frozen_sha256'] == 'ed72c22c9d16e13d455f83af09bda05420ad5e81d678c39a800bd721a550204d'
            resolved = json.loads((run/'resolved-server-args.json').read_text())
            assert resolved['enable_deterministic_inference'] == (variant=='aligned')
            assert resolved['tp_size']==8 and resolved['ep_size']==4
            configs[variant] = resolved
            summaries[variant] = summary
        base, aligned = summaries['baseline'], summaries['aligned']
        pairs.append(dict(pair=pair, baseline=base, aligned=aligned,
                          latency_overhead_pct=(aligned['median_s']/base['median_s']-1)*100,
                          throughput_change_pct=(base['median_s']/aligned['median_s']-1)*100))
    diffs = {key: {'baseline':configs['baseline'].get(key),'aligned':configs['aligned'].get(key)}
             for key in configs['baseline'].keys()|configs['aligned'].keys()
             if configs['baseline'].get(key)!=configs['aligned'].get(key)}
    result = dict(pairs=pairs, mean_paired_latency_overhead_pct=statistics.mean(p['latency_overhead_pct'] for p in pairs),
                  independent_runs_per_variant=3, formal_batches_per_variant=60,
                  resolved_config_differences=diffs,
                  limitations=['SGLang Engine.generate only, not e2e RL', 'configured baseline on same modified stack',
                               'fixed lengths but generated identities may differ', 'three independent runs; descriptive, not a significance claim'])
    (RUNS/'inference-comparison-20260910.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
