"""Data transformations."""

import torch
from numpy.random import MT19937, Generator
from torch import Tensor


class NormalizeAction:
    """
    Normalize 3D+ tensor with given max and min values.

    Parameters
    ----------
    max_array : list[int]
        Maximum values for each dimension.
    min_array : list[int]
        Minimum values for each dimension.
    """

    def __init__(self, max_array: list[int], min_array: list[int]) -> None:
        self.max_array = Tensor(max_array)
        self.min_array = Tensor(min_array)

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply normalization.

        Parameters
        ----------
        data : Tensor
            Data to normalize.
            Shape: [B*, len(self.max_array)].

        Returns
        -------
        Tensor
            Normalized data. Shape: [B*, len(self.max_array)].

        """
        copy_data = data.detach().clone()
        copy_data -= self.min_array
        copy_data *= 1.0 / (self.max_array - self.min_array)
        copy_data *= 2.0
        copy_data += -1.0
        return copy_data


class RandomWindow:
    """シーケンスデータをランダムにウィンドウで切り出す."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.randgen = Generator(MT19937(42))

    def __call__(self, data: Tensor) -> Tensor:
        """Select start idx with `randgen2` and slice data."""
        seq_len = data.shape[0]
        start_idx = self.randgen.integers(0, seq_len - self.window_size)
        return data[start_idx : start_idx + self.window_size]


class RemoveDim:
    """指定した次元を削除する."""

    def __init__(self, axis: int, indices_to_remove: list[int]) -> None:
        self.axis = axis
        self.remove = indices_to_remove

    def __call__(self, data: Tensor) -> Tensor:
        """Erase specified dimension."""
        all_indices = list(range(data.size(self.axis)))
        keep = [i for i in all_indices if i not in self.remove]
        return torch.index_select(data, self.axis, torch.tensor(keep))
