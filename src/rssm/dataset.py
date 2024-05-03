"""Datamodule that reads local `.pt` files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from rssm.custom_types import DataGroup


class EpisodeDataset(Dataset[DataGroup]):
    """Dataset with actions & observations."""

    def __init__(
        self,
        *,
        action_path_list: list[Path],
        observation_path_list: list[Path],
        action_transform: Compose | None = None,
        observation_transform: Compose | None = None,
        action_augmentation: Compose | None = None,
        observation_augmentation: Compose | None = None,
    ) -> None:
        """Initialize `PlayDataset` ."""
        super().__init__()
        self.act_path_list = action_path_list
        self.obs_path_list = observation_path_list
        self.act_transform = action_transform or Compose([])
        self.obs_transform = observation_transform or Compose([])
        self.act_augmentation = action_augmentation or Compose([])
        self.obs_augmentation = observation_augmentation or Compose([])

    def __len__(self) -> int:
        """Return the number of data."""
        return len(self.act_path_list)

    def __getitem__(self, idx: int) -> DataGroup:
        """Apply transforms to and return tensors."""
        act = self.act_transform(load_tensor(self.act_path_list[idx]))
        obs = self.obs_transform(load_tensor(self.obs_path_list[idx]))
        return (
            self.act_augmentation(act[:-1]),
            self.obs_augmentation(obs[:-1]),
            act[1:],
            obs[1:],
        )


def load_tensor(path: Path) -> Tensor:
    """`.npy`/`.pt`ファイルを読み込み, `torch.Tensor`に変換する."""
    if path.suffix == ".npy":
        return Tensor(np.load(path))
    if path.suffix == ".pt" and isinstance(tensor := torch.load(path), Tensor):
        return tensor
    msg = f"Unknown file extension: {path.suffix}"
    raise ValueError(msg)


class EpisodeDataModule(LightningDataModule):
    """DataModule with actions & observations."""

    def __init__(
        self,
        *,
        data_name: str,
        batch_size: int,
        num_workers: int,
        action_transform: Compose | None = None,
        observation_transform: Compose | None = None,
        action_augmentation: Compose | None = None,
        observation_augmentation: Compose | None = None,
    ) -> None:
        """Initialize variables."""
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.action_transform = action_transform
        self.observation_transform = observation_transform
        self.action_augmentation = action_augmentation
        self.observation_augmentation = observation_augmentation

        self.path_to_train = Path("data") / data_name / "train"
        self.path_to_val = Path("data") / data_name / "validation"

    def setup(self, stage: str = "train") -> None:  # noqa: ARG002
        """Set up train/val/test dataset."""
        self.train_dataset = EpisodeDataset(
            action_path_list=list(self.path_to_train.glob("act*.pt")),
            observation_path_list=list(self.path_to_train.glob("obs*.pt")),
            action_transform=self.action_transform,
            observation_transform=self.observation_transform,
            action_augmentation=self.action_augmentation,
            observation_augmentation=self.observation_augmentation,
        )
        self.val_dataset = EpisodeDataset(
            action_path_list=list(self.path_to_val.glob("act*.pt")),
            observation_path_list=list(self.path_to_val.glob("obs*.pt")),
            action_transform=self.action_transform,
            observation_transform=self.observation_transform,
            action_augmentation=None,
            observation_augmentation=None,
        )

    def train_dataloader(self) -> DataLoader[DataGroup]:
        """Define training dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=1,
        )

    def val_dataloader(self) -> DataLoader[DataGroup]:
        """Define validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=1,
        )