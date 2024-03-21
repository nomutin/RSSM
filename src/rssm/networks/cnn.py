"""
CNN-based encoder and decoder for the observation model.

References
----------
- https://github.com/williamFalcon/pytorch-lightning-vae

"""

from __future__ import annotations

from einops import pack, unpack
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
from torch import Tensor, nn


class Encoder(nn.Module):
    """
    Observatino Encoder.

    Input: [B, C, H, W]
    -> Resnet 18 Encoder: [B, 512 * H // 32 * W // 32]
    -> Linear: [B, obs_embed_size]
    """

    def __init__(
        self,
        obs_embed_size: int,
        obs_shape: tuple[int, int, int],
    ) -> None:
        """Set parameters and build model."""
        super().__init__()
        self.model = nn.Sequential(
            resnet18_encoder(first_conv=True, maxpool1=False),
            nn.Linear(obs_shape[1] * obs_shape[2] // 2, obs_embed_size),
        )

    def forward(self, observations: Tensor) -> Tensor:
        """
        Forward method.

        The batch shape is consistent at the input and output.
        """
        observations, ps = pack([observations], "* c h w")
        feature = self.model.forward(observations)
        return unpack(feature, ps, "* d")[0]


class Decoder(nn.Module):
    """
    Observation Decoder.

    Input: [B, latent_dim]
    -> Resnet 18 Decoder: [B, C, H, W]
    """

    def __init__(
        self,
        latent_size: int,
        obs_shape: tuple[int, int, int],
    ) -> None:
        """Set parameters and build model."""
        super().__init__()
        self.model = nn.Sequential(
            resnet18_decoder(
                latent_dim=latent_size,
                input_height=obs_shape[1],
                first_conv=True,
                maxpool1=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward method.

        The batch shape is consistent at the input and output.
        """
        features, ps = pack([features], "* d")
        reconstruction = self.model.forward(features)
        return unpack(reconstruction, ps, "* c h w")[0]
