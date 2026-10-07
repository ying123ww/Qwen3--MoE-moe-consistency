"""Adapt the existing exclusive analyzer to these uncompressed isolated traces."""
import builtins
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
TOOLS = ROOT.parent.parent / 'gpu_tools'
sys.path.insert(0, str(TOOLS))
import analyze_optimizer_exclusive as analyzer
# Adapter only: these traces use the same annotations, but are uncompressed JSON.
analyzer.gzip = SimpleNamespace(open=builtins.open)
result = json.loads((ROOT / 'result.json').read_text()) if (ROOT / 'result.json').exists() else {'rows': []}
rows = []
for variant in ('old', 'pin_only', 'event_only', 'both'):
    row = analyzer.analyze(ROOT / (variant + '.json'))
    row['variant'] = variant
    es = json.loads((ROOT / (variant + '.json')).read_text())['traceEvents']
    span = next(e for e in es if e.get('name') == 'P1/optimizer_step')
    lo, hi = span['ts'], span['ts'] + span['dur']
    cpu = [e for e in es if e.get('ph') == 'X' and e.get('dur',0)>0 and (e.get('pid'),e.get('tid')) == (span['pid'],span['tid']) and e.get('cat') in ('cpu_op','cuda_runtime','cuda_driver') and lo <= e['ts'] < hi]
    parts = analyzer.exclusive(cpu,lo,hi,lambda e:e['cat']+'/'+e['name'])
    h_intervals = [(e['ts'],e['ts']+e['dur']) for e in es if e.get('cat') == 'gpu_memcpy' and 'HtoD' in e['name']]
    row['h2d_overlap_exclusive_by_op_s'] = {k:analyzer.total(analyzer.intersection(v,h_intervals))/1e6 for k,v in parts.items() if analyzer.total(analyzer.intersection(v,h_intervals))>0}
    row['exclusive_by_op_s'] = {k:analyzer.total(v)/1e6 for k,v in parts.items()}

    row['exclusive_cpu_math_s'] = sum(v for k,v in row['exclusive_main_thread_s'].items() if k.startswith('math/'))
    h2d = row['gpu_observed_overlap']['H2D']
    row['h2d_math_overlap_fraction'] = h2d['overlap_cpu_math_s'] / h2d['union_s']
    runs = [r for r in result['rows'] if r['variant'] == variant]
    row['unprofiled_rep_medians_s'] = [r['median_s'] for r in runs]
    row['unprofiled_median_of_rep_medians_s'] = statistics.median(row['unprofiled_rep_medians_s']) if runs else None
    row['unprofiled_pooled_median_s'] = statistics.median([t for r in runs for t in r['step_s']]) if runs else None
    rows.append(row)
output = {
    'method': 'Exclusive main-thread wall accounting: CUDA API priority, otherwise innermost CPU operation. H2D GPU intervals are linked by correlation to APIs inside P1/optimizer_step and intersected with exclusive CPU math intervals. Profiler traces are single isolated steps; speed comparison uses separate synchronized unprofiled timings. CPU op intervals include worker waiting and are not CPU core utilization. Observed interval overlap is not a causal speedup estimate. Synthetic gradients and 16 parameters are not full training; no claim about the historical 15.4 second optimizer.',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), TOOLS/'analyze_optimizer_exclusive.py', TOOLS/'analyze_native_routepack_trace.py', ROOT/'result.json') if p.exists()},
    'environment_note': 'Runs 01 and 02 inherited CUDA_LAUNCH_BLOCKING=1, confirmed by the coordinating agent from the remote environment; their zero overlap is not a valid test of asynchronous overlap recovery. Run 03 is explicitly rerun without launch blocking; refer to its result configuration.',
    'results': rows,
}
(ROOT/'exclusive_trace_analysis.json').write_text(json.dumps(output, indent=2)+'\n')
for r in rows:
    h=r['gpu_observed_overlap']['H2D']; parts=r['exclusive_main_thread_s']
    print(r['variant'], 'profiled wall ms', r['optimizer_wall_s']*1000, 'CPU math ms',r['exclusive_cpu_math_s']*1000,'H2D API ms',parts.get('api/H2D',0)*1000,'H2D GPU ms',h['union_s']*1000,'overlap ms',h['overlap_cpu_math_s']*1000, 'overlap%',r['h2d_math_overlap_fraction']*100,'unprofiled median ms',(r['unprofiled_median_of_rep_medians_s'] or 0)*1000)
