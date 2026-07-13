from typing import Optional, Sequence
import torch

from dict_hub import PositiveTripleIndex
from doc import Example


_POSITIVE_INDEX: Optional[PositiveTripleIndex] = None


def set_positive_index(index: PositiveTripleIndex):
    global _POSITIVE_INDEX
    _POSITIVE_INDEX = index


def construct_mask(row_examples: Sequence[Example], col_examples: Sequence[Example]) -> torch.Tensor:
    rows, cols = len(row_examples), len(col_examples)
    mask = torch.ones(rows, cols, dtype=torch.bool)
    if _POSITIVE_INDEX is None:
        return mask

    for i, row in enumerate(row_examples):
        known = _POSITIVE_INDEX.get_tails(row.head_id, row.relation)
        if not known:
            continue
        for j, col in enumerate(col_examples):
            if i == j and rows == cols:
                continue
            if col.tail_id in known:
                mask[i, j] = False
    return mask


def construct_self_negative_mask(examples: Sequence[Example]) -> torch.Tensor:
    if _POSITIVE_INDEX is None:
        return torch.ones(len(examples), dtype=torch.bool)
    values = [
        not _POSITIVE_INDEX.is_positive(ex.head_id, ex.relation, ex.head_id)
        for ex in examples
    ]
    return torch.tensor(values, dtype=torch.bool)
