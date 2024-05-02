# ruff: noqa: SLF001
"""Callbacks for RSSM v1."""

from typing import TYPE_CHECKING

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from rssm.utils.visualize import to_pca_wandb_image, to_wandb_movie
from rssm.v1.module import RSSMV1

if TYPE_CHECKING:
    from torch import Tensor


class LogRSSMV1Output(Callback):
    """Callback to visualize RSSM output."""

    def __init__(self, every_n_epochs: int, indices: list[int]) -> None:
        """Set parameters and load dataloader."""
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        if not isinstance(rssm := pl_module, RSSMV1):
            return
        if not isinstance(d := trainer.validate_loop._data_source, DataLoader):
            return
        for batch in d:
            outputs: dict[str, Tensor] = rssm.predict_step(batch)

        posterior = outputs["posterior"]
        posterior_recon = outputs["posterior_recon"]
        prior_recon = outputs["prior_recon"]

        logger.experiment.log(
            {
                "posterior": to_wandb_movie(posterior_recon[self.indices]),
                "prior": to_wandb_movie(prior_recon[self.indices]),
                "deter": to_pca_wandb_image(posterior.deter, self.indices),
                "stoch": to_pca_wandb_image(posterior.stoch, self.indices),
            },
        )
