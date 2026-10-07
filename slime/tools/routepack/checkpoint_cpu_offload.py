"""Opt-in saved-hidden-tensor CPU storage for the isolated schedule benchmark.

Only floating nonleaf [T,1,H] CUDA tensors are selected. Parameters (leaf),
2-D weights and vocabulary logits are excluded. Copies are synchronous and
preserve values/dtype; their cost remains inside the measured training step.
"""
from contextlib import contextmanager


def selected_hidden(tensor, hidden_size, *, simulate_cpu=False):
    return (tensor.device.type == ('cpu' if simulate_cpu else 'cuda') and
            tensor.is_floating_point() and not tensor.is_leaf and tensor.ndim == 3 and
            tensor.shape[1] == 1 and tensor.shape[2] == hidden_size)


@contextmanager
def saved_hidden_cpu_offload(hidden_size, *, simulate_cpu=False):
    import torch
    stats = dict(pack_count=0, unpack_count=0, packed_bytes=0,
                 selector='floating nonleaf [T,1,hidden_size]; CUDA only',
                 synchronous_copies=True)

    def pack(tensor):
        if not selected_hidden(tensor, hidden_size, simulate_cpu=simulate_cpu):
            return (False, tensor.detach())
        cpu = tensor.detach().to(device='cpu', copy=True, non_blocking=False)
        stats['pack_count'] += 1
        stats['packed_bytes'] += tensor.numel() * tensor.element_size()
        return (True, cpu, tensor.device)

    def unpack(payload):
        if not payload[0]:
            return payload[1]
        stats['unpack_count'] += 1
        # CPU simulation deliberately allocates on both legs, so the exact
        # value-preservation path is exercised without CUDA availability.
        return payload[1].to(device=payload[2], copy=True, non_blocking=False)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats


@contextmanager
def saved_softmax_cpu_offload(ppo_module, vocab_size, chunk_size, *, protected_storages=(), simulate_cpu=False):
    """Offload only save slot 0 of the single vocab-logprob autograd Function.

    _calculate_log_probs_and_entropy_chunk invokes exactly one Function.apply;
    its first saved tensor is log_prob_softmax. All remaining saved slots,
    including logits, are untouched even if their shape matches.
    """
    import torch
    original = ppo_module._calculate_log_probs_and_entropy_chunk
    protected = set(protected_storages)
    stats = dict(pack_count=0, unpack_count=0, packed_bytes=0, chunk_calls=0,
                 selector='save slot 0 in vocab-logprob chunk Function; FP32 [chunk,V]')

    def wrapped(*args, **kwargs):
        position = 0
        stats['chunk_calls'] += 1
        def pack(tensor):
            nonlocal position
            slot = position
            position += 1
            selected = (slot == 0 and tensor.device.type == ('cpu' if simulate_cpu else 'cuda') and
                        tensor.dtype == torch.float32 and tensor.ndim == 2 and
                        0 < tensor.shape[0] <= chunk_size and tensor.shape[1] == vocab_size and
                        tensor.untyped_storage().data_ptr() not in protected)
            if not selected:
                return (False, tensor.detach())
            cpu = tensor.detach().to('cpu', copy=True, non_blocking=False)
            stats['pack_count'] += 1
            stats['packed_bytes'] += tensor.numel() * tensor.element_size()
            return (True, cpu, tensor.device)
        def unpack(payload):
            if not payload[0]:
                return payload[1]
            stats['unpack_count'] += 1
            return payload[1].to(payload[2], copy=True, non_blocking=False)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return original(*args, **kwargs)

    ppo_module._calculate_log_probs_and_entropy_chunk = wrapped
    try:
        yield stats
    finally:
        ppo_module._calculate_log_probs_and_entropy_chunk = original
