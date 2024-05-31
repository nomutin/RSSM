# ruff: noqa: SLF001
"""Callbacks for RSSM."""

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from rssm.base.module import RSSM
from rssm.custom_types import DataGroup
from rssm.state import cat_states
from rssm.utils.visualize import to_wandb_movie


def get_validation_data(trainer: Trainer) -> DataGroup:
    """Get **validation** data from trainer."""
    data_module = trainer.validate_loop._data_source.instance
    if not isinstance(data_module, LightningDataModule):
        msg = "DataModule not found."
        raise TypeError(msg)
    loader = data_module.val_dataloader()
    return next(iter(loader))  # type: ignore[no-any-return]


class LogRSSMOutput(Callback):
    """Callback to visualize RSSM output."""

    def __init__(
        self,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        """Set parameters and load dataloader."""
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices
        self.query_length = query_length
        self.fps = fps

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log RSSM imagination outputs."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        if not isinstance(rssm := pl_module, RSSM):
            return

        batch = get_validation_data(trainer)
        action_input, observation_input, _, observation_target = (
            tensor[self.indices].to(rssm.device) for tensor in batch
        )
        posterior, _ = rssm.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=rssm.initial_state(observation_input[:, 0]),
        )
        posterior_recon = rssm.decoder.forward(posterior.feature)
        prior = rssm.rollout_transition(
            actions=action_input[:, self.query_length :],
            prev_state=posterior[:, self.query_length - 1],
        )
        prior = cat_states([posterior[:, : self.query_length], prior], dim=1)
        prior_recon = rssm.decoder.forward(prior.feature)
        logger.experiment.log(
            {
                "observation": to_wandb_movie(observation_target, self.fps),
                "recon#posterior": to_wandb_movie(posterior_recon, self.fps),
                "recon#prior": to_wandb_movie(prior_recon, self.fps),
            },
        )
