"""Datamodule that reads local `.pt` files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from .augmentation import Transforms


class ActionObservationDataset(Dataset):
    """Dataset with actions & observations."""

    def __init__(
        self,
        path_to_data: Path,
        seq_per_batch: int,
        transforms: Transforms,
    ) -> None:
        """Initialize `PlayDataset` ."""
        super().__init__()
        self.path_to_data = path_to_data
        self.seq_per_batch = seq_per_batch
        self.transforms = transforms
        self.data_size = len(list(self.path_to_data.glob("action_*.pt")))

    def __len__(self) -> int:
        """Return the number of data increase by `seq_per_batch`."""
        return self.data_size * self.seq_per_batch

    def load_action(self, idx: int) -> Tensor:
        """Load action data (sequence)."""
        action_path = self.path_to_data / f"action_{idx}.pt"
        return torch.load(action_path)

    def load_observation(self, idx: int) -> Tensor:
        """Load observation data (sequence)."""
        observation_path = self.path_to_data / f"observation_{idx}.pt"
        return torch.load(observation_path)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply transforms to and return tensors."""
        idx = idx // self.seq_per_batch
        action = self.load_action(idx)
        observation = self.load_observation(idx)
        return (
            self.transforms.action_input(action[:-1]),
            self.transforms.observation_input(observation[:-1]),
            action[1:],
            observation[:-1],
        )


class ActionObservationDataModule(lightning.LightningDataModule):
    """DataModule with actions & observations."""

    def __init__(  # noqa: PLR0913
        self,
        data_name: str,
        batch_size: int,
        num_workers: int,
        seq_per_batch: int,
        train_transforms: Transforms,
        val_transforms: Transforms,
    ) -> None:
        """Initialize variables."""
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_per_batch = seq_per_batch
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self.path_to_train = Path("data") / data_name / "train"
        self.path_to_val = Path("data") / data_name / "validation"

    def setup(self, stage: str = "train") -> None:  # noqa: ARG002
        """Set up train/val/test dataset."""
        self.train_dataset = ActionObservationDataset(
            path_to_data=self.path_to_train,
            seq_per_batch=self.seq_per_batch,
            transforms=self.train_transforms,
        )
        self.val_dataset = ActionObservationDataset(
            path_to_data=self.path_to_val,
            seq_per_batch=1,
            transforms=self.val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        """Define training dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Define validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_dataset.data_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
