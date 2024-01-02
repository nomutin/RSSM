"""Self-contained programs that works on the Hook."""

from __future__ import annotations

import lightning
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBarTheme,
)
from lightning.pytorch.loggers import WandbLogger


class ProgressBarCallback(RichProgressBar):
    """
    Make the progress bar richer.

    References
    ----------
    * https://qiita.com/akihironitta/items/edfd6b29dfb67b17fb00
    """

    def __init__(self) -> None:
        """Rich progress bar with custom theme."""
        theme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
        super().__init__(theme=theme)


class LogStateDictCallBack(lightning.Callback):
    """Callback to log model in wandb."""

    def __init__(self, every_n_epochs: int) -> None:
        """Set parameters and load dataloader."""
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Log model to wandb every `every_n_epochs` epochs."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return
        trainer.logger.experiment.log(pl_module.state_dict())
