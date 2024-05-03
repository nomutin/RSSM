# ruff: noqa: SLF001
"""Callbacks for RSSM."""

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from rssm.base.module import RSSM
from rssm.custom_types import DataGroup
from rssm.state import cat_states
from rssm.utils.visualize import to_pca_wandb_image, to_wandb_movie


def get_validation_data(trainer: Trainer) -> DataGroup:
    """Get **validation** data from trainer."""
    loader = trainer.validate_loop._data_source.instance
    if not isinstance(loader, DataLoader):
        msg = "DataLoader not found."
        raise TypeError(msg)
    return next(iter(loader))  # type: ignore[no-any-return]


class LogRSSMOutput(Callback):
    """Callback to visualize RSSM output."""

    def __init__(
        self,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
    ) -> None:
        """Set parameters and load dataloader."""
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices
        self.query_length = query_length

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
        action_input, observation_input, _, observation_target = batch
        posterior, _ = rssm.rollout_representation(
            actions=action_input[self.indices],
            observations=observation_input[self.indices],
            prev_state=rssm.initial_state(batch_size=len(self.indices)),
        )
        posterior_recon = rssm.decoder.forward(
            state=posterior.stoch,
            rnn_hidden=posterior.deter,
        )
        prior = rssm.rollout_transition(
            actions=action_input[:, : self.query_length],
            prev_state=posterior[:, -1],
        )
        prior = cat_states([posterior[:, : self.query_length], prior], dim=1)
        prior_recon = rssm.decoder.forward(
            state=prior.stoch,
            rnn_hidden=prior.deter,
        )
        logger.experiment.log(
            {
                "observation": to_wandb_movie(observation_target),
                "recon#posterior": to_wandb_movie(posterior_recon),
                "recon#prior": to_wandb_movie(prior_recon),
                "pca#deter": to_pca_wandb_image(posterior.deter, self.indices),
                "pca#stoch": to_pca_wandb_image(posterior.stoch, self.indices),
            },
        )
