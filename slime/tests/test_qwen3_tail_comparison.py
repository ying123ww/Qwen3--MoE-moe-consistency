from pathlib import Path

import torch

from slime.utils.compare_qwen3_tail import (
    TrainTailSequence,
    compare_final_norm,
)


def _write_sglang_tail_dump(
    path: Path,
    rank: int,
    final_norm: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True)
    local_slice = slice(rank * 2, (rank + 1) * 2)
    torch.save(
        {
            "model.forward_batch_info.input_ids": torch.tensor([10, 11, 12, 13]),
            "model.forward_batch_info.positions": torch.tensor([0, 1, 2, 3]),
            "model.forward_batch_info.rids": ["rollout-0"],
            "model.forward_batch_info.extend_seq_lens": torch.tensor([4]),
            "model.norm": final_norm[local_slice],
        },
        path,
    )


def test_final_norm_comparison_maps_attention_tp_token_rows(tmp_path):
    final_norm = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
    train_sequence = TrainTailSequence(
        tokens=torch.tensor([10, 11, 12, 13, 14]),
        final_norm=final_norm,
        source="train",
    )
    dump_files = []
    for rank in range(2):
        dump_file = (
            tmp_path
            / f"TP{rank}_PP0_Rank{rank}_pid1"
            / "Pass00000.pt"
        )
        _write_sglang_tail_dump(dump_file, rank, final_norm)
        dump_files.append(dump_file)

    stats = compare_final_norm(
        dump_files,
        [train_sequence],
        {"rollout-0": 0},
        attn_tp_size=2,
    )

    assert stats["tokens"] == 4
    assert stats["max_abs"] == 0.0
    assert stats["mean_abs"] == 0.0
    assert stats["exact"]
    assert stats["mismatched_elements"] == 0
