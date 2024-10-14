"""Reccurent State Space Model (RSSM)."""

from distribution_extension import kl_divergence
from lightning import LightningModule
from torch import Tensor, nn

from rssm.networks import Representation, Transition
from rssm.objective import likelihood
from rssm.state import State, stack_states


class RSSM(LightningModule):
    """
    Reccurent State Space Model (RSSM).

    References
    ----------
    * https://arxiv.org/abs/1912.01603 [Hafner+ 2019]
    * https://arxiv.org/abs/2010.02193 [Hafner+ 2021]
    * https://github.com/juliusfrost/dreamer-pytorch

    Parameters
    ----------
    representation : Representation
        Representation model (Approx. Posterior).
    transition : Transition
        Transition model (Prior).
    encoder : nn.Module
        Observation encoder.
        I/O: [*B, C, H, W] -> [*B, obs_embed_size].
    decoder : nn.Module
        Observation decoder.
        I/O: [*B, obs_embed_size] -> [*B, C, H, W].
    init_proj : nn.Module
        Initial projection layer.
        I/O: [*B, obs_embed_size] -> [*B, deterministic_size].
    kl_coeff : float
        KL Divergence coefficient.
    use_kl_balancing : bool
        Whether to use KL balancing.

    """

    def __init__(
        self,
        *,
        representation: Representation,
        transition: Transition,
        encoder: nn.Module,
        decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.transition = transition
        self.encoder = encoder
        self.decoder = decoder
        self.init_proj = init_proj
        self.kl_coeff = kl_coeff
        self.use_kl_balancing = use_kl_balancing

    def initial_state(self, observation: Tensor) -> State:
        """
        Generate initial state as zero matrix.

        Parameters
        ----------
        observation : Tensor
            4D Tensor [batch_size, channel, height, width].

        Returns
        -------
        State
            Initial state.
        """
        obs_embed = self.encoder(observation)
        deter = self.init_proj(obs_embed)
        stoch = self.transition.rnn_to_prior_projector(deter)
        distribution = self.representation.distribution_factory(stoch)
        return State(deter=deter, distribution=distribution).to(self.device)

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

        Returns
        -------
        tuple[State, State]
            Posterior and prior states.
            shape: [batch_size, seq_len, state_size].

        """
        obs_embed = self.encoder.forward(observations)
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

    def rollout_transition(self, actions: Tensor, prev_state: State) -> State:
        """
        Rollout transition (prior loop) aka latent imagination.

        Parameters
        ----------
        actions : Tensor
            3D Tensor [batch_size, seq_len, action_size].
        prev_state : State
            2D Parameters [batch_size, state_size].

        Returns
        -------
        State
            Prior states [batch_size, seq_len, state_size].

        """
        priors = []
        for t in range(actions.shape[1]):
            prev_state = self.transition.forward(actions[:, t], prev_state)
            priors.append(prev_state)
        return stack_states(priors, dim=1)

    def training_step(self, batch: tuple[Tensor, ...], _: int) -> dict[str, Tensor]:
        """
        Rollout training step.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            Batch data.

        Returns
        -------
        dict[str, Tensor]
            Losses.
        """
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: tuple[Tensor, ...], _batch_index: int) -> dict[str, Tensor]:
        """
        Rollout validation step.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            Batch data.
        _batch_index : int
            Batch index.

        Returns
        -------
        dict[str, Tensor]
            Losses. Each loss is prefixed with "val_".
        """
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """
        Rollout common step for training and validation.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            Batch data.
            action_input : Tensor
                3D Tensor [batch_size, seq_len, action_size].
            observation_input : Tensor
                5D Tensor [batch_size, seq_len, channel, height, width].
            action_target : Tensor
                3D Tensor [batch_size, seq_len, action_size].
            observation_target : Tensor
                5D Tensor [batch_size, seq_len, channel, height, width].

        Returns
        -------
        dict[str, Tensor]
            Losses.
        """
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
            use_balancing=self.use_kl_balancing,
        ).mul(self.kl_coeff)
        return {
            "loss": recon_loss + kl_div,
            "recon": recon_loss,
            "kl": kl_div,
        }
