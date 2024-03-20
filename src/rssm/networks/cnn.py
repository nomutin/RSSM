"""
CNN-based encoder and decoder for the observation model.

References
----------
- https://github.com/williamFalcon/pytorch-lightning-vae


TODO: サイズ可変に
TODO: Decoderの出力をsigmoidにする

"""

from __future__ import annotations

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
from torch import nn


class Encoder(nn.Sequential):
    """
    Observatino Encoder.

    Input: [B, C, H, W]
    -> Resnet 50 Encoder: [B, 512 * H // 32 * W // 32]
    -> Linear: [B, latent_dim * 2]
    -> Normal: [B, latent_dim] (mean), [B, latent_dim] (stddev)
    """

    def __init__(
        self,
        latent_dim: int,
        obs_shape: tuple[int, int, int],
    ) -> None:
        """Set parameters and build model."""
        super().__init__()
        conv = resnet18_encoder(first_conv=True, maxpool1=False)
        lin = nn.Linear(obs_shape[1] * obs_shape[2] // 2, latent_dim * 2)
        super().__init__(conv, lin)


class Decoder(nn.Sequential):
    """
    Observation Decoder.

    Input: [B, latent_dim]
    -> Resnet 50 Decoder: [B, C, H, W]
    """

    def __init__(
        self,
        latent_dim: int,
        obs_shape: tuple[int, int, int],
    ) -> None:
        """Set parameters and build model."""
        conv = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=obs_shape[1],
            first_conv=True,
            maxpool1=False,
        )
        super().__init__(conv)
