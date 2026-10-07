"""CPU protocol tests for opt-in collection and exact monkey-patch restoration."""
from contextlib import contextmanager
import tempfile
from types import SimpleNamespace
import unittest
from gpu_component_profile import component_profile


class Module:
    def forward(self):
        return 7


class FakeProfiler:
    def __init__(self):
        self.toggles=[]
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def toggle_collection_dynamic(self,enabled,activities):self.toggles.append(enabled)
    def export_chrome_trace(self,path):
        with open(path,'w') as f:f.write('{}')
    def key_averages(self):return []


class Tests(unittest.TestCase):
    def run_case(self, fail=False):
        profilers=[];sync=[];ranges=[]
        def profile(**kwargs):
            result=FakeProfiler();profilers.append(result);return result
        @contextmanager
        def record(name):
            ranges.append(name)
            yield
        torch=SimpleNamespace(profiler=SimpleNamespace(profile=profile,
            ProfilerActivity=SimpleNamespace(CPU='cpu',CUDA='cuda'),record_function=record),
            cuda=SimpleNamespace(synchronize=lambda:sync.append(1)))
        mlp=Module();mlp.experts=Module();mlp.experts.linear_fc1=Module()
        saved_closure=lambda:mlp.experts.linear_fc1.forward()
        mlp.experts.forward=saved_closure # existing instance override must survive exactly
        model=Module();model.named_modules=lambda:[('decoder.layers.0.mlp',mlp)]
        model.forward=lambda:mlp.experts.forward()
        original_model=model.forward
        schedules=SimpleNamespace(backward_step=lambda:mlp.experts.forward())
        original_backward=schedules.backward_step
        optimizer=SimpleNamespace(step=lambda:3);original_step=optimizer.step
        with tempfile.TemporaryDirectory() as tmp:
            trace=component_profile([model],optimizer,tmp,0,torch_module=torch,schedules_module=schedules,aligned_module=SimpleNamespace())
            try:
                with trace:
                    self.assertEqual(model.forward(),7)
                    self.assertEqual(model.forward(),7)
                    schedules.backward_step();schedules.backward_step();optimizer.step()
                    if fail:raise ValueError('injected')
            except ValueError:
                self.assertTrue(fail)
            self.assertIs(model.forward,original_model)
            self.assertIs(mlp.experts.forward,saved_closure)
            self.assertNotIn('forward',vars(mlp.experts.linear_fc1))
            self.assertIs(schedules.backward_step,original_backward)
            self.assertIs(optimizer.step,original_step)
            self.assertEqual(len(profilers),3)
            self.assertTrue(all(not p.toggles for p in profilers))
            self.assertEqual([p['phase'] for p in trace.results()['phases']], ['forward','backward','optimizer'])
            self.assertEqual(len(sync),6)
            self.assertEqual(trace.results()['failed'],fail)
            self.assertEqual(trace.results()['calls'],dict(forward=2,backward=2,optimizer=1))
            self.assertEqual(len([r for r in ranges if 'native_TE_' in r]),2)
    def test_normal_restore(self):self.run_case()
    def test_exception_restore(self):self.run_case(True)


if __name__=='__main__':unittest.main()
