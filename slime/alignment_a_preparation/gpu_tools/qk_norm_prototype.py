"""Standalone opt-in prototypes; not connected to the model until exact-gated."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm_body(X, W, Y, row, HEADS: tl.constexpr, H: tl.constexpr,
               STRIDE_M: tl.constexpr, STRIDE_D: tl.constexpr, EPS,
               BLOCK: tl.constexpr):
    token = row // HEADS
    head = row % HEADS
    input_base = X + token * STRIDE_M + head * H * STRIDE_D
    output_base = Y + row * H
    # Keep the upstream 1024-wide masked reduction and arithmetic order.
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for offset in range(0, H, BLOCK):
        columns = offset + tl.arange(0, BLOCK)
        mask = columns < H
        vals = tl.load(input_base + columns * STRIDE_D, mask=mask, other=0.0)
        fp32 = vals.to(tl.float32)
        squares = fp32 * fp32
        sum_sq += tl.sum(tl.where(mask, squares, 0.0))
    mean_sq = sum_sq / H
    rms = tl.sqrt(mean_sq + EPS)
    inv_rms = 1.0 / rms
    for offset in range(0, H, BLOCK):
        columns = offset + tl.arange(0, BLOCK)
        mask = columns < H
        vals = tl.load(input_base + columns * STRIDE_D, mask=mask, other=0.0)
        weight = tl.load(W + columns, mask=mask, other=1.0)
        output = (vals.to(tl.float32) * inv_rms * weight.to(tl.float32)).to(vals.dtype)
        tl.store(output_base + columns, output, mask=mask)


@triton.jit
def _strided_norm(X, W, Y, HEADS: tl.constexpr, H: tl.constexpr,
                  STRIDE_M: tl.constexpr, STRIDE_D: tl.constexpr, EPS,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    _norm_body(X, W, Y, row, HEADS, H, STRIDE_M, STRIDE_D, EPS, BLOCK)


@triton.jit
def _joint_norm(Q, K, QW, KW, QO, KO, Q_ROWS: tl.constexpr,
                Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, H: tl.constexpr,
                Q_STRIDE_M: tl.constexpr, Q_STRIDE_D: tl.constexpr,
                K_STRIDE_M: tl.constexpr, K_STRIDE_D: tl.constexpr,
                Q_EPS, K_EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if row < Q_ROWS:
        _norm_body(Q, QW, QO, row, Q_HEADS, H, Q_STRIDE_M, Q_STRIDE_D, Q_EPS, BLOCK)
    else:
        _norm_body(K, KW, KO, row - Q_ROWS, K_HEADS, H, K_STRIDE_M, K_STRIDE_D, K_EPS, BLOCK)


def forward(q, k, q_weight, k_weight, head_dim, q_eps, k_eps, mode='joint'):
    if mode not in ('strided', 'joint'):
        raise ValueError('Unknown standalone Q/K prototype mode')
    if not (q.is_cuda and q.device == k.device == q_weight.device == k_weight.device
            and q.ndim == k.ndim == 2 and q.shape[0] == k.shape[0]
            and 1 <= q.shape[0] <= 4 and head_dim == 128
            and q.dtype == k.dtype == torch.bfloat16
            and q.stride(1) == k.stride(1) == 1
            and q.shape[1] % head_dim == k.shape[1] % head_dim == 0
            and q_weight.shape == k_weight.shape == (head_dim,)
            and q_weight.is_contiguous() and k_weight.is_contiguous()
            and q_weight.dtype == k_weight.dtype
            and q_weight.dtype in (torch.bfloat16, torch.float32)
            and k_weight.dtype in (torch.bfloat16, torch.float32)):
        raise ValueError('Standalone prototype only supports the captured small-M BF16/head-128 case')
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    q_heads, k_heads = q.shape[1] // head_dim, k.shape[1] // head_dim
    q_rows, k_rows = q.shape[0] * q_heads, k.shape[0] * k_heads
    if mode == 'strided':
        for x, w, y, heads, rows, eps in ((q, q_weight, q_out, q_heads, q_rows, q_eps),
                                        (k, k_weight, k_out, k_heads, k_rows, k_eps)):
            _strided_norm[(rows,)](x, w, y, heads, head_dim, x.stride(0), x.stride(1), eps,
                                   BLOCK=1024, num_warps=4, num_stages=3)
    else:
        _joint_norm[(q_rows + k_rows,)](q, k, q_weight, k_weight, q_out, k_out, q_rows,
            q_heads, k_heads, head_dim, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            q_eps, k_eps, BLOCK=1024, num_warps=4, num_stages=3)
    return q_out, k_out
