import copy
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from summarize_gpu_comparison import summarize_run, compare_runs


def fixture():
    return {'passed':True,'scope':'fixture only','checkpoint_cpu_offload':False,'measured_repeats':2,
      'checks':[dict(rank=r,weights_unchanged=True,gradient_check=dict(finite=True,nonzero=True)) for r in range(2)],
      'iterations':[[dict(rank=r,iteration=i,phase=phase,full_step_s=t,optimizer_invocations=1)
       for r,t in enumerate(times)] for i,(phase,times) in enumerate([
        ('warmup',[99,99]),('measured',[1,3]),('measured',[5,2]),('attribution',[99,99]),('gradient_validation',[99,99])])]}


class SummaryTest(unittest.TestCase):
    def test_max_then_median_excludes_instrumented(self):
        out=summarize_run(fixture(),2)
        self.assertEqual(out['p50_max_rank_full_step_s'],4)
        self.assertEqual([r['max_full_step_s'] for r in out['raw_measured_iterations']],[3,5])

    def test_relative_change(self):
        a=summarize_run(fixture(),2);b=copy.deepcopy(a);b['p50_max_rank_full_step_s']=3
        out=compare_runs({'native':a,'routepack':b})
        self.assertEqual(out['runs']['routepack']['p50_time_change_vs_native_percent'],-25)

    def test_cli_native_routepack_without_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for method in ('native','routepack'):
                (root/(method+'.json')).write_text(json.dumps(fixture()))
            output=root/'summary.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_gpu_comparison.py')),
                '--native',str(root/'native.json'),'--routepack',str(root/'routepack.json'),
                '--expected-ranks','2','--output',str(output)],check=True,capture_output=True,text=True)
            result=json.loads(output.read_text())
            self.assertEqual(set(result['runs']),{'native','routepack'})
            self.assertEqual(result['methods_without_supplied_timing'],['random'])
            self.assertNotIn('random',result['runs'])

    def test_reject_invalid(self):
        mutations=[lambda d:d.update(passed=False),lambda d:d['iterations'][1].pop(),
          lambda d:d['iterations'][1][1].update(rank=0),lambda d:d['iterations'][1][0].update(full_step_s=float('nan')),
          lambda d:d['iterations'][1][0].update(phase='attribution'),lambda d:d.update(measured_repeats=3),
          lambda d:d['checks'][0].update(weights_unchanged=False)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                d=fixture();mutate(d)
                with self.assertRaises(ValueError):summarize_run(d,2)


if __name__=='__main__':unittest.main()
