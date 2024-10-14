"""Callbacks for RSSM."""

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from rssm.core import RSSM
from rssm.state import cat_states


class LogRSSMOutput(Callback):
    """Callback to visualize RSSM output."""

    def __init__(
        self,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices
        self.query_length = query_length
        self.fps = fps

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log RSSM imagination outputs."""
        if trainer.current_epoch % self.every_n_epochs != 0 or trainer.current_epoch == 0:
            return
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        if not isinstance(rssm := pl_module, RSSM):
            return

        for stage in ("train", "val"):
            dataloader = getattr(trainer.datamodule, f"{stage}_dataloader")()  # type: ignore[attr-defined]
            action_input, observation_input, _, observation_target = (
                tensor[self.indices].to(rssm.device) for tensor in next(iter(dataloader))
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

            self.log_video(observation_target, f"observation({stage})", logger)
            self.log_video(posterior_recon, f"recon#posterior({stage})", logger)
            self.log_video(prior_recon, f"recon#prior({stage})", logger)

    def log_video(self, batch_video: Tensor, key: str, logger: WandbLogger) -> None:
        """Log video to wandb."""
        logger.log_video(key=key, videos=list(batch_video.cpu().mul(255)), fps=[self.fps] * len(self.indices))
