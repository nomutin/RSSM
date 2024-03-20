"""World Model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from distribution_extension import kl_divergence

from rssm.base.loss import likelihood
from rssm.base.module import RSSM
from rssm.base.state import State, cat_states, stack_states
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
            latent_dim=obs_embed_size,
            obs_shape=observation_shape,
        )
        self.decoder = Decoder(
            latent_dim=obs_embed_size,
            obs_shape=observation_shape,
        )
        self.kl_coeff = kl_coeff

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
        ).mul(self.kl_coeff)
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
