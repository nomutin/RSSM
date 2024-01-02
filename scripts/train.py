"""Script to train a model."""

from __future__ import annotations

from pathlib import Path

import click
import lightning
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from rssm.utils.config import load_config


@click.command()
@click.argument("config_path", type=str)
@click.option("--dev", is_flag=True, default=False)
def train(config_path: str | Path, dev: bool) -> None:  # noqa: FBT001
    """Train a model."""
    lightning.seed_everything(42)
    config_path = Path(config_path)
    config = load_config(config_path)
    model = instantiate(config.model)
    datamodule = instantiate(config=config.datamodule)
    callbacks = [instantiate(config=cb) for cb in config.callbacks]
    if dev:
        trainer = instantiate(
            config=config.trainer,
            fast_dev_run=True,
            callbacks=callbacks,
        )
    else:
        Path("wandb").mkdir(exist_ok=True)
        logger = WandbLogger(project=config_path.stem, save_dir="./wandb/")
        log_model = ModelCheckpoint(
            dirpath=logger.experiment.dir,
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(log_model)
        trainer = instantiate(
            config=config.trainer,
            logger=logger,
            fast_dev_run=False,
            callbacks=callbacks,
        )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
