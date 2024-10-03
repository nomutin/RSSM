"""Datamodule that reads local `.pt` files."""

from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


def load_tensor(path: Path) -> Tensor:
    """
    Load tensor from file(.npy, .pt).

    Parameters
    ----------
    path : Path
        File path.

    Returns
    -------
    Tensor
        Loaded tensor.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    if path.suffix == ".npy":
        return torch.Tensor(np.load(path))
    if path.suffix == ".pt" and isinstance(tensor := torch.load(path, weights_only=False), Tensor):
        return tensor
    msg = f"Unknown file extension: {path.suffix}"
    raise ValueError(msg)


def split_path_list(path_list: list[Path], train_ratio: float) -> tuple[list[Path], list[Path]]:
    """
    Split the path list into train and test.

    Parameters
    ----------
    path_list : list[Path]
        List of file paths.
    train_ratio : float
        Ratio of train data.

    Returns
    -------
    tuple[list[Path], list[Path]]
        Train and test path list.
    """
    split_point = int(len(path_list) * train_ratio)
    return path_list[:split_point], path_list[split_point:]


class EpisodeDataset(Dataset[tuple[Tensor, ...]]):
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

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        """Apply transforms to and return tensors."""
        act = self.act_transform(load_tensor(self.act_path_list[idx]))
        obs = self.obs_transform(load_tensor(self.obs_path_list[idx]))
        return (
            self.act_augmentation(act[:-1]),
            self.obs_augmentation(obs[:-1]),
            act[1:],
            obs[1:],
        )


class EpisodeDataModule(LightningDataModule):
    """DataModule with actions & observations."""

    def __init__(
        self,
        *,
        data_name: str,
        batch_size: int,
        num_workers: int,
        action_preprocess: Compose | None = None,
        observation_preprocess: Compose | None = None,
        action_transform: Compose | None = None,
        observation_transform: Compose | None = None,
        action_augmentation: Compose | None = None,
        observation_augmentation: Compose | None = None,
    ) -> None:
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.action_preprocess = action_preprocess or Compose([])
        self.observation_preprocess = observation_preprocess or Compose([])
        self.action_transform = action_transform
        self.observation_transform = observation_transform
        self.action_augmentation = action_augmentation
        self.observation_augmentation = observation_augmentation

        self.path_to_data = Path("data") / data_name

    def prepare_data(self) -> None:
        """Save processed data to temporary directory."""
        for act_path in sorted(self.path_to_data.glob("act*")):
            act = self.action_preprocess(load_tensor(act_path))
            new_path = Path("tmp") / f"{act_path.stem}.pt"
            torch.save(act.clone(), new_path)

        for obs_path in sorted(self.path_to_data.glob("obs*")):
            obs = self.observation_preprocess(load_tensor(obs_path))
            new_path = Path("tmp") / f"{obs_path.stem}.pt"
            torch.save(obs.clone(), new_path)

    def setup(self, stage: str = "train") -> None:
        """Set up train/val/test dataset."""
        act_path_list = sorted(Path("tmp").glob("act*"))
        obs_path_list = sorted(Path("tmp").glob("obs*"))
        train_act_list, val_act_list = split_path_list(act_path_list, 0.8)
        train_obs_list, val_obs_list = split_path_list(obs_path_list, 0.8)

        self.train_dataset = EpisodeDataset(
            action_path_list=train_act_list,
            observation_path_list=train_obs_list,
            action_transform=self.action_transform,
            observation_transform=self.observation_transform,
            action_augmentation=self.action_augmentation,
            observation_augmentation=self.observation_augmentation,
        )
        self.val_dataset = EpisodeDataset(
            action_path_list=val_act_list,
            observation_path_list=val_obs_list,
            action_transform=self.action_transform,
            observation_transform=self.observation_transform,
            action_augmentation=None,
            observation_augmentation=None,
        )

        if stage == "test":
            self.test_dataset = EpisodeDataset(
                action_path_list=train_act_list + val_act_list,
                observation_path_list=train_obs_list + val_obs_list,
                action_transform=self.action_transform,
                observation_transform=self.observation_transform,
                action_augmentation=None,
                observation_augmentation=None,
            )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        Define training dataloader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            Training dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        Define validation dataloader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            Validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )
