"""Data visualization utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from einops import pack, unpack
from matplotlib import figure
from torch import Tensor, uint8
from wandb import Image, Video


def visualize_2d_data(
    data: Tensor,
    indices: list[int],
    x_label: str,
    y_label: str,
) -> figure.Figure:
    """Visualize 2D data."""
    fig, axe = plt.subplots(figsize=(8, 8), tight_layout=True)
    for idx in indices:
        axe.plot(data[idx, :, 0], data[idx, :, 1], alpha=0.5)
    axe.set_xlabel(xlabel=x_label)
    axe.set_ylabel(ylabel=y_label)
    return fig


def pca(data: Tensor, n_components: int = 2) -> tuple[Tensor, Tensor]:
    """
    Apply PCA on 2D+ Tensor.

    References
    ----------
    * https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html

    Returns
    -------
    Tensor
        PCA-transformed data. Tensor shaped [batch*, n_components].
    Tensor
        Explained variance ratio. Tensor shaped [n_components].

    """
    data, ps = pack([data], "* d")
    _, s, v = torch.pca_lowrank(data, q=n_components)
    [data_pca] = unpack(torch.matmul(data, v), ps, "* d")
    ratio = (s**2) / (data.shape[0] - 1) / data.var(dim=0).sum()
    return data_pca, ratio


def to_wandb_images(tensors: Tensor) -> list[Image]:
    """Convert batched image tensor to wandb images."""
    tensors = tensors.detach().cpu()
    if tensors.dtype == uint8:
        tensors = tensors.float() / 255
    return [Image(tensor) for tensor in tensors]


def to_pca_wandb_image(tensor: Tensor, indices: list[int]) -> Image:
    """Apply PCA on 2D+ Tensor and convert to wandb image."""
    tensor_pca, variance_ratio = pca(tensor.detach().cpu())
    fig = visualize_2d_data(
        data=tensor_pca,
        indices=indices,
        x_label=f"PC1({variance_ratio[0]:.2f})",
        y_label=f"PC2({variance_ratio[1]:.2f})",
    )
    return Image(fig)


def to_wandb_movie(tensor: Tensor) -> Video:
    """Convert image tensor to wandb video."""
    tensor = tensor.detach().cpu().mul(255)
    return Video(tensor.to(dtype=uint8))  # type: ignore
