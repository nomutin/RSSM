"""Discrete Reccurent State Space Model(RSSM V2)."""

from cnn import Decoder, DecoderConfig, Encoder, EncoderConfig
from distribution_extension import MultiOneHotFactory, kl_divergence
from torch import Tensor, nn

from rssm.base.module import RSSM
from rssm.custom_types import DataGroup, LossDict
from rssm.objective import likelihood
from rssm.state import State
from rssm.v2.network import RepresentationV2, TransitionV2


class RSSMV2(RSSM):
    """
    Categorical Reccurent State Space Model(RSSM).

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
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
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
        self.encoder = Encoder(config=encoder_config)
        self.decoder = Decoder(config=decoder_config)
        self.distribution_factory = MultiOneHotFactory(
            class_size=class_size,
            category_size=category_size,
        )

        self.init_proj = nn.Linear(obs_embed_size, deterministic_size)
        self.kl_coeff = kl_coeff

    def initial_state(self, observation: Tensor) -> State:
        """Generate initial state as zero matrix."""
        obs_embed = self.encoder(observation)
        deter = self.init_proj(obs_embed)
        stoch = self.transition.rnn_to_prior_projector(deter)
        distribution = self.representation.distribution_factory.forward(stoch)
        return State(deter=deter, distribution=distribution).to(self.device)

    def shared_step(self, batch: DataGroup) -> LossDict:
        """Rollout common step for training and validation."""
        action_input, observation_input, _, observation_target = batch
        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=self.initial_state(observation_input[:, 0]),
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
            use_balancing=True,
        ).mul(self.kl_coeff)
        return {
            "loss": recon_loss + kl_div,
            "recon": recon_loss,
            "kl": kl_div,
        }
