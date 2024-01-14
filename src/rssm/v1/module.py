"""World Model."""

from __future__ import annotations

from typing import Any

import torch
from cnn import Decoder, DecoderConfig, Encoder, EncoderConfig
from distribution_extention import NormalFactory, kl_divergence
from torch import Tensor

from rssm.base.loss import likelihood
from rssm.base.module import RSSM, RepresentationConfig, TransitionConfig
from rssm.base.state import State, cat_states, stack_states
from rssm.v1.representation import RepresentationV1
from rssm.v1.transition import TransitionV1


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
        transition_config: TransitionConfig,
        representation_config: RepresentationConfig,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
    ) -> None:
        """Initialize RSSM components."""
        super().__init__()
        self.save_hyperparameters()
        self.representation = RepresentationV1(representation_config)
        self.transition = TransitionV1(transition_config)
        self.encoder = Encoder(config=encoder_config)
        self.decoder = Decoder(config=decoder_config)
        self.distribution_factory = NormalFactory()
        self.kl_factor = 0.1

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        deter_size = self.hparams["representation_config"].deterministic_size
        stoch_size = self.hparams["representation_config"].stochastic_size
        deter = torch.zeros([batch_size, deter_size])
        stoch = torch.zeros([batch_size, stoch_size * 2])
        distribution = self.distribution_factory.forward(stoch)
        return State(deter=deter, distribution=distribution).to(self.device)

    def encode(self, observation: Tensor) -> Tensor:
        """Encode observation."""
        return self.encoder.forward(observation)

    def decode(self, state: State) -> Tensor:
        """Decode state."""
        return self.decoder.forward(state.feature)

    def rollout_representation(
        self,
        actions: Tensor,
        observations: Tensor,
        prev_state: State,
    ) -> tuple[State, State]:
        """
        Rollout representation (posterior loop).

        Parameters
        ----------
        actions : Tensor
            3D Tensor [batch_size, seq_len, action_size].
        observations : Tensor
            5D Tensor [batch_size, seq_len, channel, height, width].
        prev_state : State
            2D Parameters [batch_size, state_size].
        """
        obs_embed = self.encode(observation=observations)
        priors, posteriors = [], []
        for t in range(observations.shape[1]):
            prior = self.transition.forward(actions[:, t], prev_state)
            posterior = self.representation.forward(obs_embed[:, t], prior)
            priors.append(prior)
            posteriors.append(posterior)
            prev_state = posterior

        prior = stack_states(priors, dim=1)
        posterior = stack_states(posteriors, dim=1)
        return posterior, prior

    def rollout_transition(
        self,
        actions: Tensor,
        prev_state: State,
    ) -> State:
        """
        Rollout transition (prior loop) aka latent imagination.

        Parameters
        ----------
        actions : Tensor
            3D Tensor [batch_size, seq_len, action_size].
        prev_state : State
            2D Parameters [batch_size, state_size].
        """
        priors = []
        for t in range(actions.shape[1]):
            prev_state = self.transition.forward(actions[:, t], prev_state)
            priors.append(prev_state)
        return stack_states(priors, dim=1)

    def _shared_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        action_input, observation_input, _, observation_target = batch
        batch_size = action_input.shape[0]
        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=self.initial_state(batch_size=batch_size),
        )
        reconstruction = self.decode(state=posterior)
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
            "variance_max": posterior.distribution.variance.max().detach(),
            "variance_min": posterior.distribution.variance.min().detach(),
        }

    def test_step(self, batch: tuple[Tensor, ...]) -> dict[str, Any]:
        """Run test step."""
        action_input, observation_input, _, _ = batch
        batch_size = action_input.shape[0]
        initial_state = self.initial_state(batch_size=batch_size)
        posterior, _ = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=initial_state,
        )
        prior = self.rollout_transition(
            actions=action_input[:, 5:],
            prev_state=posterior[:, 5],
        )
        prior = cat_states([posterior[:, :5], prior], dim=1)
        posterior_recon = self.decode(state=posterior)
        prior_recon = self.decode(state=prior)
        return {
            "posterior": posterior,
            "prior": prior,
            "posterior_recon": posterior_recon,
            "prior_recon": prior_recon,
        }
