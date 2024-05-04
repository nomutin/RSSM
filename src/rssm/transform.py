"""Data transformations."""

import torch
from numpy.random import MT19937, Generator
from torch import Tensor


class NormalizeAction:
    """
    行動データの正規化.

    torchvision等と同様, `data = (data - mean) / std` の計算を行う.

    Parameters
    ----------
    mean : list[float]
        平均値. `(max + min) / 2` で求める.
    std : list[float]
        標準偏差. `(max - min) / 2` で求める.

    """

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = Tensor(mean)
        self.std = Tensor(std)

    def __call__(self, data: Tensor) -> Tensor:
        """
        正規化.

        Parameters
        ----------
        data : Tensor
            正規化するデータ.
            最後の次元数が `len(mean)` と一致してればバッチサイズは何でもいい
        """
        return data.sub(self.mean).div(self.std)


class RandomWindow:
    """シーケンスデータをランダムにウィンドウで切り出す."""

    def __init__(self, window_size: int) -> None:
        """Initialize parameters."""
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
        """Initialize parameters."""
        self.axis = axis
        self.remove = indices_to_remove

    def __call__(self, data: Tensor) -> Tensor:
        """Erase specified dimension."""
        all_indices = list(range(data.size(self.axis)))
        keep = [i for i in all_indices if i not in self.remove]
        return torch.index_select(data, self.axis, torch.tensor(keep))
