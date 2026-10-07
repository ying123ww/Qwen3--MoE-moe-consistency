import unittest
import torch
from torch.utils.checkpoint import checkpoint
from routepack.checkpoint_cpu_offload import selected_hidden, saved_hidden_cpu_offload


class Offload(unittest.TestCase):
    def test_selection_excludes_weights_logits_and_leaf_tensors(self):
        leaf = torch.randn(5, 1, 8, requires_grad=True)
        hidden = leaf * 1.2
        self.assertFalse(selected_hidden(hidden, 8))
        self.assertTrue(selected_hidden(hidden, 8, simulate_cpu=True))
        self.assertFalse(selected_hidden(leaf, 8, simulate_cpu=True))
        self.assertFalse(selected_hidden(hidden.squeeze(1), 8, simulate_cpu=True))
        self.assertFalse(selected_hidden(hidden, 100, simulate_cpu=True))
        self.assertFalse(selected_hidden(hidden.expand(5, 2, 8), 8, simulate_cpu=True))
        self.assertFalse(selected_hidden(hidden.to(torch.int32), 8, simulate_cpu=True))

    def test_checkpoint_output_and_gradients_exact_with_cpu_copy_roundtrip(self):
        torch.manual_seed(7)
        inputs = torch.randn(7, 1, 8)
        weights = torch.randn(8, 8)
        def run(offload):
            x, w = inputs.clone().requires_grad_(), weights.clone().requires_grad_()
            def block(z): return (z @ w).tanh()
            from contextlib import nullcontext
            ctx = saved_hidden_cpu_offload(8, simulate_cpu=True) if offload else nullcontext(None)
            with ctx as stats:
                y = checkpoint(block, x * 1.25, use_reentrant=True)
                y = checkpoint(block, y, use_reentrant=True)
                y.square().sum().backward()
            return y.detach(), x.grad, w.grad, stats
        reference = run(False)
        copied = run(True)
        for left, right in zip(reference[:3], copied[:3]):
            self.assertTrue(torch.equal(left, right))
        self.assertGreater(copied[3]['pack_count'], 0)
        self.assertGreater(copied[3]['unpack_count'], 0)
        self.assertGreater(copied[3]['packed_bytes'], 0)


class SoftmaxOffload(unittest.TestCase):
    def test_exact_saved_slot_and_function_restoration(self):
        from types import SimpleNamespace
        from routepack.checkpoint_cpu_offload import saved_softmax_cpu_offload
        class Toy(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                softmax = x.softmax(-1)
                ctx.save_for_backward(softmax, x)  # same shape, slot 1 must stay untouched
                return softmax.square().sum()
            @staticmethod
            def backward(ctx, grad):
                softmax, x = ctx.saved_tensors
                return 2 * softmax * (softmax - softmax.square().sum(-1, keepdim=True)) * grad
        original = Toy.apply
        module = SimpleNamespace(_calculate_log_probs_and_entropy_chunk=original)
        torch.manual_seed(4)
        value = torch.randn(3, 7)
        left = value.clone().requires_grad_()
        ref = original(left); ref.backward()
        right = value.clone().requires_grad_()
        with saved_softmax_cpu_offload(module, 7, 4, simulate_cpu=True) as stats:
            out = module._calculate_log_probs_and_entropy_chunk(right)
            out.backward()
        self.assertIs(module._calculate_log_probs_and_entropy_chunk, original)
        self.assertEqual(stats['pack_count'], 1)
        self.assertEqual(stats['packed_bytes'], 3 * 7 * 4)
        self.assertTrue(torch.equal(ref, out))
        self.assertTrue(torch.equal(left.grad, right.grad))

    def test_real_vocab_logprob_operator_output_and_gradient_exact(self):
        import importlib.util
        from pathlib import Path
        from routepack.checkpoint_cpu_offload import saved_softmax_cpu_offload
        source = Path(__file__).resolve().parents[2] / 'slime/utils/ppo_utils.py'
        spec = importlib.util.spec_from_file_location('routepack_cpu_test_ppo', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(11)
        data = torch.randn(11, 17)
        targets = torch.randint(17, (11,))
        def run(offload):
            from contextlib import nullcontext
            x = data.clone().requires_grad_()
            ctx = saved_softmax_cpu_offload(module, 17, 3, simulate_cpu=True) if offload else nullcontext(None)
            with ctx as stats:
                lp, entropy = module.calculate_log_probs_and_entropy(
                    x, targets, None, with_entropy=True, with_entropy_grad=False, chunk_size=3)
                lp.sum().backward()
            return lp.detach(), entropy.detach(), x.grad, stats
        reference, copied = run(False), run(True)
        for left, right in zip(reference[:3], copied[:3]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(copied[3]['pack_count'], 4)
        self.assertEqual(copied[3]['packed_bytes'], 11 * 17 * 4)

    def test_protected_storage_and_exception_restore(self):
        from types import SimpleNamespace
        from routepack.checkpoint_cpu_offload import saved_softmax_cpu_offload
        class SaveInput(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sum()
            @staticmethod
            def backward(ctx, grad):
                return torch.ones_like(ctx.saved_tensors[0]) * grad
        original = SaveInput.apply
        module = SimpleNamespace(_calculate_log_probs_and_entropy_chunk=original)
        x = torch.randn(3, 7, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with saved_softmax_cpu_offload(module, 7, 4,
                    protected_storages=[x.untyped_storage().data_ptr()], simulate_cpu=True) as stats:
                module._calculate_log_probs_and_entropy_chunk(x).backward()
                self.assertEqual(stats['pack_count'], 0)
                raise RuntimeError('test failure')
        self.assertIs(module._calculate_log_probs_and_entropy_chunk, original)


if __name__ == '__main__':
    unittest.main()
