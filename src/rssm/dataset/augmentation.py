"""Data Augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributions as td
from numpy.random import MT19937, Generator
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Compose

empty_compose = Compose([])


@dataclass
class Transforms:
    """Transforms for `PlayDataset` ."""

    action: Compose = empty_compose
    observation: Compose = empty_compose
    action_input: Compose = empty_compose
    observation_input: Compose = empty_compose


class AdditiveNoise:
    """
    Add zero-mean Gaussian noise.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    """

    def __init__(self, std: float = 0.1) -> None:
        """Initialize parameters."""
        self.std = std

    def __call__(self, data: Tensor) -> Tensor:
        """Add noise to data."""
        max_, min_, device = data.max(), data.min(), data.device
        noise = torch.normal(mean=0, std=self.std, size=data.shape)
        return torch.clamp(data + noise.to(device), min_, max_)


class MultiplicativeNoise:
    """
    Add zero-mean Uniform noise.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    """

    def __init__(self, low: float = 0.8, high: float = 1.2) -> None:
        """Initialize parameters."""
        self.low = low
        self.high = high

    def __call__(self, data: Tensor) -> Tensor:
        """Add noise to data."""
        epsilon = td.Uniform(low=self.low, high=self.high).sample(data.shape)
        return data * epsilon.to(data.device)


class DynamicsNoise:
    """
    Apply state mix-up.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    * [mixup](https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float = 0.4) -> None:
        """Initialize parameters and random generator."""
        self.alpha = alpha
        self.randgen = Generator(MT19937(42))

    def __call__(self, data: Tensor) -> Tensor:
        """Apply state mix-up."""
        eps_size = [data.shape[0], *[1] * (data.ndim - 1)]
        eps_array = self.randgen.beta(self.alpha, self.alpha, size=eps_size)
        eps_tensor = torch.from_numpy(eps_array).float()
        shift = torch.cat([data[1:].clone(), data[-1:].clone()], dim=0)
        return (eps_tensor * data + (1 - eps_tensor) * shift).to(data.device)


class RandomWindow:
    """Randomly select a window of sequence."""

    def __init__(self, lower_window_size: int, upper_window_size: int) -> None:
        """Initialize parameters."""
        self.lower = lower_window_size
        self.upper = upper_window_size
        self.randgen1 = Generator(MT19937(42))
        self.randgen2 = Generator(MT19937(42))
        self._update_window_size()

    def _update_window_size(self) -> None:
        """Update window size with `randgen1` ."""
        self.window_size = self.randgen1.integers(self.lower, self.upper)

    def __call__(self, data: Tensor) -> Tensor:
        """Select start idx with `randgen2` and slice data."""
        self._update_window_size()
        seq_len = data.shape[0]
        start_idx = self.randgen2.integers(0, seq_len - self.window_size)
        return data[start_idx : start_idx + self.window_size]


class TorchvisionTransform:
    """Apply `torchvision` transform to movie."""

    def __init__(self, transform_name: str, **params: Any) -> None:
        """Initialize parameters."""
        self.transform = getattr(transforms, transform_name)(**params)

    def __call__(self, movie: Tensor) -> Tensor:
        """Apply transform to movie."""
        return torch.stack([self.transform(image) for image in movie])
