import torch
from distribution_extension import Normal, kl_divergence
from torch import Tensor
from torchrl.modules import ObsDecoder, ObsEncoder

from rssm.base.module import RSSM
from rssm.base.state import State, cat_states
from rssm.objective import likelihood
from rssm.v1.network import RepresentationV1, TransitionV1


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
        *,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        obs_embed_size: int,
        action_size: int,
        activation_name: str,
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
        self.encoder = ObsEncoder(num_layers=3)
        self.decoder = ObsDecoder(num_layers=3)
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size
        self.kl_coeff = kl_coeff

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        deter = torch.zeros([batch_size, self.deterministic_size])
        stoch = torch.zeros([batch_size, self.stochastic_size * 2])
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
            "variance_max": posterior.distribution.variance.max().detach(),
            "variance_min": posterior.distribution.variance.min().detach(),
        }

    def imagination_step(
        self,
        batch: list[Tensor],
        query_length: int,
    ):
        action_input, observation_input, _, observation_target = batch
        batch_size = action_input.shape[0]
        posterior, _ = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=self.initial_state(batch_size=batch_size),
        )
        posterior_reconstruction = self.decoder.forward(
            state=posterior.stoch,
            rnn_hidden=posterior.deter,
        )
        prior = self.rollout_transition(
            actions=action_input[:, :query_length],
            prev_state=posterior[:, -1],
        )
        prior = cat_states([posterior[:, :query_length], prior], dim=1)
        prior_reconstruction = self.decoder.forward(
            state=prior.stoch,
            rnn_hidden=prior.deter,
        )
        return {
            "posterior_reconstruction": posterior_reconstruction,
            "prior_reconstruction": prior_reconstruction,
            "observation_target": observation_target,
        }
