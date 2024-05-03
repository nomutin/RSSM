"""Abstract classes for RSSM."""

import tempfile
from pathlib import Path

import torch
import wandb
from lightning import LightningModule
from torch import Tensor, nn

from rssm.base.network import Representation, Transition
from rssm.custom_types import DataGroup, LossDict
from rssm.state import State, stack_states


class RSSM(LightningModule):
    """
    Abstract class for RSSM.

    Note:
    ----
    Inherited classes must implement:
        - `initial_state()`
        - `_shared_step()`

    """

    def __init__(self) -> None:
        super().__init__()
        self.representation = Representation()
        self.transition = Transition()
        self.encoder = nn.Module()
        self.decoder = nn.Module()

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        raise NotImplementedError

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

        """
        priors = []
        for t in range(actions.shape[1]):
            prev_state = self.transition.forward(actions[:, t], prev_state)
            priors.append(prev_state)
        return stack_states(priors, dim=1)

    def training_step(self, batch: DataGroup, **_: str) -> LossDict:
        """Rollout training step."""
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: DataGroup, _: int) -> LossDict:
        """Rollout validation step."""
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def shared_step(self, batch: DataGroup) -> LossDict:
        """Rollout common step for training and validation."""
        raise NotImplementedError

    @classmethod
    def load_from_wandb(cls, reference: str) -> "RSSM":
        """Load the model from wandb checkpoint."""
        run = wandb.Api().artifact(reference)  # type: ignore[no-untyped-call]
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(run.download(root=tmpdir))
            model = cls.load_from_checkpoint(
                checkpoint_path=ckpt / "model.ckpt",
                map_location=torch.device("cpu"),
            )
        if not isinstance(model, cls):
            msg = f"Model is not an instance of {cls}"
            raise TypeError(msg)
        return model
