"""World Model."""

from typing import TYPE_CHECKING

import torch
from distribution_extension import Normal, kl_divergence

from rssm.base.loss import likelihood
from rssm.base.module import RSSM
from rssm.base.state import State
from rssm.networks.cnn import Decoder, Encoder
from rssm.v1.representation import RepresentationV1
from rssm.v1.transition import TransitionV1

if TYPE_CHECKING:
    from torch import Tensor


class RSSMV1(RSSM):
    """
    Continuous reccurent State Space Model(RSSM).

    References
    ----------
    - https://arxiv.org/abs/1912.01603 [Hafner+ 2019]
    - https://github.com/juliusfrost/dreamer-pytorch

    """

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
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
        self.representation = RepresentationV1(
            deterministic_size=deterministic_size,
            stochastic_size=stochastic_size,
            hidden_size=hidden_size,
            obs_embed_size=obs_embed_size,
            activation_name=activation_name,
        )
        self.transition = TransitionV1(
            deterministic_size=deterministic_size,
            stochastic_size=stochastic_size,
            hidden_size=hidden_size,
            action_size=action_size,
            activation_name=activation_name,
        )
        self.encoder = Encoder(
            obs_embed_size=obs_embed_size,
            obs_shape=observation_shape,
        )
        self.decoder = Decoder(
            latent_size=deterministic_size + stochastic_size,
            obs_shape=observation_shape,
        )
        self.kl_coeff = kl_coeff

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        deter_size = self.hparams["representation_config"].deterministic_size
        stoch_size = self.hparams["representation_config"].stochastic_size
        deter = torch.zeros([batch_size, deter_size])
        stoch = torch.zeros([batch_size, stoch_size * 2])
        mean, std = torch.chunk(stoch, 2, dim=-1)
        distribution = Normal(mean, std)
        return State(deter=deter, distribution=distribution).to(self.device)

    def _shared_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        action_input, observation_input, _, observation_target = batch
        batch_size = action_input.shape[0]
        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=self.initial_state(batch_size=batch_size),
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
        ).mul(self.kl_coeff)
        return {
            "loss": recon_loss + kl_div,
            "recon": recon_loss,
            "kl": kl_div,
            "variance_max": posterior.distribution.variance.max().detach(),
            "variance_min": posterior.distribution.variance.min().detach(),
        }
