"""Abstract classes for RSSM."""


from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import lightning
import torch
from torch import nn, optim

import wandb

if TYPE_CHECKING:
    from torch import Tensor

    from rssm.base.state import State


@dataclass
class RepresentationConfig:
    """Parameters for RSSM Representation Model."""

    obs_embed_size: int
    deterministic_size: int
    stochastic_size: int
    hidden_size: int
    activation_name: str
    category_size: int = 0
    class_size: int = 0


class Representation(nn.Module):
    """
    RSSM Representation Model.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """Single step transition, includes prior transition."""
        raise NotImplementedError


@dataclass
class TransitionConfig:
    """Parameters for RSSM Transition Model."""

    deterministic_size: int
    stochastic_size: int
    hidden_size: int
    action_size: int
    activation_name: str
    category_size: int = 0
    class_size: int = 0


class Transition(nn.Module):
    """
    RSSM Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

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

    def initial_state(self, batch_size: int) -> State:
        """Generate initial state as zero matrix."""
        raise NotImplementedError

    def configure_optimizers(self) -> optim.Optimizer:
        """Choose what optimizers to use."""
        return optim.AdamW(self.parameters(), lr=1e-3)

    def encode(self, observations: Tensor) -> Tensor:
        """Encode observations into embed tensor."""
        raise NotImplementedError

    def decode(self, state: State) -> Tensor:
        """Decode state into observations."""
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
        run = wandb.Api().run(reference)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_name, cpu = "best_model.ckpt", torch.device("cpu")
            ckpt = run.file(ckpt_name).download(replace=True, root=tmpdir)
            return cls.load_from_checkpoint(ckpt.name, map_location=cpu)
