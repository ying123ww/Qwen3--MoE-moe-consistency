import math
from types import SimpleNamespace
import unittest

from routepack.gpu_schedule_benchmark import advantage_value, optimizer_tree, install, forward_with_optional_attribution, cpu_metrics


class Helpers(unittest.TestCase):
    def test_advantages_follow_sample_identity_after_exchange(self):
        native = [9, 2, 18, 7]
        exchanged = [18, 9, 7, 2]
        reference = {sid: advantage_value(sid) for sid in native}
        self.assertEqual(reference, {sid: advantage_value(sid) for sid in exchanged})
        self.assertTrue(all(math.isfinite(x) and abs(x) >= 0.5 for x in reference.values()))
        self.assertTrue(any(x < 0 for x in reference.values()))
        self.assertTrue(any(x > 0 for x in reference.values()))

    def test_shared_suboptimizers_and_cycles_are_visited_once(self):
        leaf = SimpleNamespace()
        hybrid = SimpleNamespace(cpu_optimizers=[leaf], sub_optimizers=[leaf])
        root = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=hybrid)])
        leaf.optimizer = root
        result = list(optimizer_tree(root))
        self.assertEqual(len(result), 4)
        self.assertEqual(len({id(x) for x in result}), 4)

    def test_formal_phases_have_no_internal_sync_or_clock(self):
        for phase in ('warmup', 'measured', 'gradient_validation'):
            def forbidden():
                raise AssertionError('Unexpected internal timing boundary')
            result, elapsed = forward_with_optional_attribution(lambda x: x + 1, phase,
                                                                 forbidden, forbidden, 4)
            self.assertEqual(result, 5)
            self.assertIsNone(elapsed)

    def test_attribution_alone_brackets_forward(self):
        events = []
        times = iter([10.0, 12.5])
        def sync(): events.append('sync')
        def forward():
            events.append('forward')
            return 7
        result, elapsed = forward_with_optional_attribution(forward, 'attribution', sync, lambda: next(times))
        self.assertEqual(events, ['sync', 'forward', 'sync'])
        self.assertEqual((result, elapsed), (7, 2.5))

    def test_singular_optimizer_links_are_discovered(self):
        left, right = SimpleNamespace(), SimpleNamespace()
        root = SimpleNamespace(cpu_optimizer=left, gpu_optimizer=right)
        self.assertEqual(len(list(optimizer_tree(root))), 3)

    def test_metrics_preserve_nested_keys(self):
        self.assertEqual(cpu_metrics(({'loss': [1.2]}, 3.4)), [{'loss': [1.2]}, 3.4])

    def test_chained_property_is_not_read_for_multiple_children(self):
        class Chained:
            chained_optimizers = [SimpleNamespace(), SimpleNamespace()]
            @property
            def optimizer(self):
                raise AssertionError('Only a single chained optimizer supports this property')
        self.assertEqual(len(list(optimizer_tree(Chained()))), 3)

    def test_nonzero_learning_rate_rejected_before_cuda_import(self):
        with self.assertRaises(ValueError):
            install(SimpleNamespace(debug_train_only=True, lr=1e-6))


if __name__ == '__main__':
    unittest.main()
