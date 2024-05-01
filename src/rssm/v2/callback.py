"""Callbacks for RSSM v2."""

from collections.abc import Mapping
from typing import Any

import lightning
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from rssm.utils.visualize import to_pca_wandb_image, to_wandb_movie
from rssm.v2.module import RSSMV2


class LogRSSMV2Output(lightning.Callback):
    """Callback to visualize RSSM output."""

    def __init__(self, every_n_epochs: int, indices: list[int]) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices

    def on_validation_batch_end(
        self,
        *,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: tuple[Tensor, ...],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log result every `every_n_epochs` epochs."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return
        if not isinstance(pl_module, RSSMV2):
            return

        # test_stepはここに定義
        outputs = pl_module.test_step(batch)

        _, observation, _, _ = batch
        posterior = outputs["posterior"]
        posterior_recon = outputs["posterior_recon"]
        prior_recon = outputs["prior_recon"]

        trainer.logger.experiment.log(
            {
                "observation": to_wandb_movie(observation[self.indices]),
                "posterior": to_wandb_movie(posterior_recon[self.indices]),
                "prior": to_wandb_movie(prior_recon[self.indices]),
                "deter": to_pca_wandb_image(posterior.deter, self.indices),
            },
        )
