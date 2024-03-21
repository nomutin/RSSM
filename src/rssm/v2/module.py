"""World Model."""

from __future__ import annotations

import torch
from distribution_extension import (
    MultiDimentionalOneHotCategoricalFactory,
    kl_divergence,
)
from torch import Tensor

from rssm.base.loss import likelihood
from rssm.base.module import RSSM
from rssm.base.state import State
from rssm.networks.cnn import Decoder, Encoder
from rssm.v2.representation import RepresentationV2
from rssm.v2.transition import TransitionV2


class RSSMV2(RSSM):
    """
    Categorical reccurent State Space Model(RSSM).

    References
    ----------
    - https://arxiv.org/abs/2010.02193

    """

    def __init__(
        self,
        deterministic_size: int,
        class_size: int,
        category_size: int,
        hidden_size: int,
        obs_embed_size: int,
        action_size: int,
        activation_name: str,
        observation_shape: tuple[int, int, int],
        kl_coeff: float,
    ) -> None:
        """Initialize RSSM components."""
        super().__init__()
        self.save_hyperparameters()
        self.representation = RepresentationV2(
            deterministic_size=deterministic_size,
            hidden_size=hidden_size,
            obs_embed_size=obs_embed_size,
            class_size=class_size,
            category_size=category_size,
            activation_name=activation_name,
        )
        self.transition = TransitionV2(
            deterministic_size=deterministic_size,
            hidden_size=hidden_size,
            action_size=action_size,
            class_size=class_size,
            category_size=category_size,
            activation_name=activation_name,
        )
        self.encoder = Encoder(
            obs_embed_size=obs_embed_size,
            obs_shape=observation_shape,
        )
        self.decoder = Decoder(
            latent_size=deterministic_size + class_size * category_size,
            obs_shape=observation_shape,
        )
        self.distribution_factory = MultiDimentionalOneHotCategoricalFactory(
            class_size=class_size,
            category_size=category_size,
        )
        self.kl_coeff = kl_coeff

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        deter_size = self.hparams["representation_config"].deterministic_size
        stoch_size = self.hparams["representation_config"].stochastic_size
        deter = torch.zeros([batch_size, deter_size])
        stoch = torch.zeros([batch_size, stoch_size])
        distribution = self.distribution_factory.forward(stoch)
        return State(deter=deter, distribution=distribution).to(self.device)

    def _shared_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        action_input, observation_input, _, observation_target = batch
        batch_size = action_input.shape[0]
        initial_state = self.initial_state(batch_size=batch_size)
        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=initial_state,
        )
        reconstruction = self.decoder.forward(posterior.feature)
        recon_loss = likelihood(
            prediction=reconstruction,
            target=observation_target,
            event_ndims=3,
        )
        kl_div = kl_divergence(
            q=posterior.distribution.independent(1),
            p=prior.distribution.independent(1),
            use_balancing=False,
        ).mul(self.kl_factor)
        return {
            "loss": recon_loss + kl_div,
            "recon": recon_loss,
            "kl": kl_div,
        }
