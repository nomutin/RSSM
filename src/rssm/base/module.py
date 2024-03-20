"""Abstract classes for RSSM."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import lightning
import torch
import wandb
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from rssm.base.state import State


class Representation(nn.Module):
    """
    RSSM Representation Model.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(self) -> None:
        """Initialize components for type hinting."""
        super().__init__()
        self.rnn_to_post_projector = nn.Module()
        self.distribution_factory = nn.Module()

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """Single step transition, includes prior transition."""
        raise NotImplementedError


class Transition(nn.Module):
    """
    RSSM Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

    def __init__(self) -> None:
        """Initialize components for type hinting."""
        super().__init__()
        self.rnn_cell = nn.Module()
        self.action_state_projector = nn.Module()
        self.rnn_to_prior_projector = nn.Module()
        self.distribution_factory = nn.Module()

    def forward(self, action: Tensor, prev_state: State) -> State:
        """
        Single step transition, includes deterministic transitions by GRUs.

        Parameters
        ----------
        action : Tensor
            (Prev) aciton of agent or robot. Shape: (batch_size, action_size)
        prev_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Prior state.

        """
        raise NotImplementedError


class RSSM(lightning.LightningModule):
    """Abstract class for RSSM."""

    def __init__(self) -> None:
        """Initialize components for type hinting."""
        super().__init__()
        self.representation = Representation()
        self.transition = Transition()

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
        Rollout posterior roop.

        Parameters
        ----------
        actions : Tensor
            Action sequence. Shape: (batch_size, seq_len, action_size)
        observations : Tensor
            Observation sequence. Shape: (batch_size, seq_len, obs_size)
        prev_state : State
            Previous state. Shape: (batch_size, feature_size)

        """
        raise NotImplementedError

    def rollout_transition(
        self,
        actions: Tensor,
        prev_state: State,
    ) -> State:
        """
        Rollout prior loop.

        Parameters
        ----------
        actions : Tensor
            Action sequence. Shape: (batch_size, seq_len, action_size)
        prev_state : State
            Previous state. Shape: (batch_size, feature_size)

        """
        raise NotImplementedError

    def training_step(self, batch: list, **_: dict) -> dict[str, Tensor]:
        """Rollout training step."""
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: list, _: int) -> dict[str, Tensor]:
        """Rollout validation step."""
        loss_dict = self._shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def _shared_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        """Rollout common step for training and validation."""
        raise NotImplementedError

    @classmethod
    def load_from_wandb(cls, reference: str) -> RSSM:
        """Load the model from wandb checkpoint."""
        run = wandb.Api().artifact(reference)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(run.download(root=tmpdir))
            return cls.load_from_checkpoint(
                checkpoint_path=ckpt / "model.ckpt",
                map_location=torch.device("cpu"),
            )
