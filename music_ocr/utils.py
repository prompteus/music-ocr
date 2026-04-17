import math
from typing import Literal

import torch
from torch import Tensor


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool,
    padding_value: float | int,
    padding_side: Literal["left", "right"],
    pad_to_multiple_of: int | None = None,
) -> Tensor:
    """
    >>> x = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    >>> pad_sequence(x, batch_first=True, padding_value=0, padding_side="right")
    tensor([[1, 2, 3],
            [4, 5, 0]])
    """

    if not sequences:
        return torch.empty(0, dtype=torch.long)

    max_len = max([s.size(0) for s in sequences])
    if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
        target_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of
    else:
        target_len = max_len

    trailing_dims = sequences[0].shape[1:] if sequences[0].ndim > 1 else ()
    if batch_first:
        out_dims = (len(sequences), target_len) + trailing_dims
    else:
        out_dims = (target_len, len(sequences)) + trailing_dims

    out_tensor = torch.full(out_dims, padding_value, dtype=sequences[0].dtype, device=sequences[0].device)

    for i, tensor in enumerate(sequences):
        length = len(tensor)
        match padding_side, batch_first:
            case "right", True:
                out_tensor[i, :length, ...] = tensor
            case "right", False:
                out_tensor[:length, i, ...] = tensor
            case "left", True:
                out_tensor[i, -length:, ...] = tensor
            case "left", False:
                out_tensor[-length:, i, ...] = tensor
            case _, _:
                raise ValueError(f"Invalid padding_side {padding_side} or batch_first {batch_first}")

    return out_tensor
