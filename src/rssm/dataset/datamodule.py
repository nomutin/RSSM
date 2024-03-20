"""Datamodule that reads local `.pt` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lightning import LightningDataModule
from torch import Tensor, load
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

empty_compose = Compose([])


@dataclass
class Transforms:
    """Transforms for `PlayDataset` ."""

    action: Compose = empty_compose
    observation: Compose = empty_compose
    action_input: Compose = empty_compose
    observation_input: Compose = empty_compose


class ActionObservationDataset(Dataset):
    """Dataset with actions & observations."""

    def __init__(self, path: Path, transforms: Transforms) -> None:
        """Initialize `PlayDataset` ."""
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.data_size = len(list(self.path.glob("action_*.pt")))

    def __len__(self) -> int:
        """Return the number of data increase by `seq_per_batch`."""
        return self.data_size

    def load_action(self, idx: int) -> Tensor:
        """Load action data (sequence)."""
        action_path = self.path / f"action_{idx}.pt"
        return load(action_path)

    def load_observation(self, idx: int) -> Tensor:
        """Load observation data (sequence)."""
        observation_path = self.path / f"observation_{idx}.pt"
        return load(observation_path)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply transforms to and return tensors."""
        action = self.transforms.action(self.load_action(idx))
        observation = self.transforms.observation(self.load_observation(idx))
        return (
            self.transforms.action_input(action[:-1]),
            self.transforms.observation_input(observation[:-1]),
            action[1:],
            observation[:-1],
        )


class ActionObservationDataModule(LightningDataModule):
    """DataModule with actions & observations."""

    def __init__(
        self,
        data_name: str,
        batch_size: int,
        num_workers: int,
        train_transforms: Transforms,
        val_transforms: Transforms,
    ) -> None:
        """Initialize variables."""
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self.path_to_train = Path("data") / data_name / "train"
        self.path_to_val = Path("data") / data_name / "validation"

    def setup(self, stage: str = "train") -> None:  # noqa: ARG002
        """Set up train/val/test dataset."""
        self.train_dataset = ActionObservationDataset(
            path=self.path_to_train,
            transforms=self.train_transforms,
        )
        self.val_dataset = ActionObservationDataset(
            path=self.path_to_val,
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
