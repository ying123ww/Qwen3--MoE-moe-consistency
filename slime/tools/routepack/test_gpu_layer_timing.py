"""CPU-only hook/event protocol tests; no torch import or GPU use."""
import os
import unittest
from unittest.mock import patch
from gpu_layer_timing import layer_timing


class Event:
    clock = 0
    def record(self):
        Event.clock += 1
        self.time = Event.clock
    def elapsed_time(self, other):
        return other.time - self.time


class Handle:
    def __init__(self, hooks, fn):
        self.hooks, self.fn = hooks, fn
    def remove(self):
        self.hooks.remove(self.fn)


class Module:
    def __init__(self, fn=lambda: None):
        self.fn, self.pre, self.post = fn, [], []
    def register_forward_pre_hook(self, fn):
        self.pre.append(fn)
        return Handle(self.pre, fn)
    def register_forward_hook(self, fn, always_call=False):
        assert always_call
        self.post.append(fn)
        return Handle(self.post, fn)
    def __call__(self):
        for fn in self.pre:
            fn(self, ())
        out = None
        try:
            out = self.fn()
            return out
        finally:
            for fn in self.post:
                fn(self, (), out)


class Model:
    def __init__(self):
        self.mlp = Module()
        self.mlp.layer_number = 3
        self.mlp.experts = Module()
        self.mlp.token_dispatcher = object()
        self.mlp.fn = self.mlp.experts
    def named_modules(self):
        return [('', self), ('decoder.layers.2.mlp', self.mlp), ('decoder.layers.2.mlp.experts', self.mlp.experts)]


class Tests(unittest.TestCase):
    def make_trace(self):
        model = Model()
        syncs = []
        trace = layer_timing(model, event_factory=Event, synchronize=lambda: syncs.append(1), rank=4)
        return model, trace, syncs

    def test_pairing_and_stage_slots(self):
        model, trace, syncs = self.make_trace()
        self.assertFalse(model.mlp.pre)
        with trace:
            for stage in ('replay_forward', 'replay_backward'):
                with patch.dict(os.environ, ROUTING_REPLAY_STAGE=stage):
                    model.mlp()
                    model.mlp()
            self.assertFalse(syncs)
            with self.assertRaises(RuntimeError):
                trace.results()
        self.assertFalse(model.mlp.pre or model.mlp.post or model.mlp.experts.pre)
        result = trace.results()
        self.assertEqual(syncs, [1])
        self.assertIs(trace.results(), result)
        rows = result['events']
        self.assertEqual([r['slot'] for r in rows], [0, 0, 1, 1, 0, 0, 1, 1])
        for moe, experts in zip(rows[::2], rows[1::2]):
            self.assertEqual(experts['parent_moe_event_id'], moe['event_id'])
            self.assertEqual((moe['layer'], moe['elapsed_ms'], experts['elapsed_ms']), (2, 3, 1))
        model.mlp()
        self.assertEqual(len(trace.rows), 8)

    def test_unobservable_and_internal_recompute(self):
        model, trace, _ = self.make_trace()
        with trace:
            model.mlp.fn = lambda: None  # A custom path bypasses experts.__call__.
            model.mlp()
            with patch.dict(os.environ, ROUTING_REPLAY_STAGE='replay_backward'):
                model.mlp.experts()  # Internal checkpoint bypasses outer MoE.
        coverage = trace.results()['coverage']
        self.assertEqual(coverage['moe_calls_without_expert_hook'], 1)
        self.assertEqual(coverage['expert_calls_without_outer_moe'], 1)

    def test_exception_cleanup(self):
        model, trace, _ = self.make_trace()
        def fail():
            raise ValueError('failure')
        model.mlp.experts.fn = fail
        with self.assertRaises(ValueError):
            with trace:
                model.mlp()
        self.assertFalse(model.mlp.pre or model.mlp.post or model.mlp.experts.pre or model.mlp.experts.post)
        self.assertTrue(trace.results()['failed'])
        self.assertTrue(all(r['elapsed_ms'] is None for r in trace.results()['events']))

    def test_missing_identity_cleans_hooks(self):
        model, trace, _ = self.make_trace()
        del model.mlp.layer_number
        with self.assertRaises(ValueError):
            with trace:
                pass
        self.assertFalse(model.mlp.pre or model.mlp.experts.pre)


if __name__ == '__main__':
    unittest.main()
