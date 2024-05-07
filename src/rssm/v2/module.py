"""Discrete Reccurent State Space Model(RSSM V2)."""

import torch
from distribution_extension import MultiOneHotFactory, kl_divergence
from torchrl.modules import ObsDecoder, ObsEncoder

from rssm.base.module import RSSM
from rssm.custom_types import DataGroup, LossDict
from rssm.objective import likelihood
from rssm.state import State
from rssm.v2.network import RepresentationV2, TransitionV2


class RSSMV2(RSSM):
    """
    Categorical reccurent State Space Model(RSSM).

    References
    ----------
    - https://arxiv.org/abs/2010.02193

    """

    def __init__(
        self,
        *,
        deterministic_size: int,
        class_size: int,
        category_size: int,
        hidden_size: int,
        obs_embed_size: int,
        action_size: int,
        activation_name: str,
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
        self.encoder = ObsEncoder(num_layers=4, channels=8)
        self.decoder = ObsDecoder(num_layers=4, channels=8)
        self.distribution_factory = MultiOneHotFactory(
            class_size=class_size,
            category_size=category_size,
        )
        self.deterministic_size = deterministic_size
        self.stochastic_size = class_size * category_size
        self.kl_coeff = kl_coeff

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        deter = torch.zeros([batch_size, self.deterministic_size])
        stoch = torch.zeros([batch_size, self.stochastic_size])
        distribution = self.distribution_factory.forward(stoch)
        return State(deter=deter, distribution=distribution).to(self.device)

    def shared_step(self, batch: DataGroup) -> LossDict:
        """Rollout common step for training and validation."""
        action_input, observation_input, _, observation_target = batch
        batch_size = action_input.shape[0]
        initial_state = self.initial_state(batch_size=batch_size)
        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=initial_state,
        )
        reconstruction = self.decoder.forward(
            state=posterior.stoch,
            rnn_hidden=posterior.deter,
        )
        recon_loss = likelihood(
            prediction=reconstruction,
            target=observation_target,
            event_ndims=3,
        )
        kl_div = kl_divergence(
            q=posterior.distribution.independent(1),
            p=prior.distribution.independent(1),
            use_balancing=False,
        ).mul(self.kl_coeff)
        return {
            "loss": recon_loss + kl_div,
            "recon": recon_loss,
            "kl": kl_div,
        }
