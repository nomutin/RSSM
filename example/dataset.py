"""Datamodule that reads local `.pt` files."""

import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import gdown
import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, StackDataset
from tqdm import tqdm

Transform: TypeAlias = Callable[[Tensor], Tensor]


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


class EpisodeDataset(Dataset[Tensor]):
    """
    Dataset for single modality data.

    Parameters
    ----------
    path_list : list[Path]
        List of file paths.
    transform : Transform
        Transform function.
    """

    def __init__(self, path_list: list[Path], transform: Transform) -> None:
        super().__init__()
        self.path_list = path_list
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of data.

        Returns
        -------
        int
            Number of data(Len of path_list).
        """
        return len(self.path_list)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Get the data at the index and apply the transform.

        Parameters
        ----------
        idx : int
            Index of the data.

        Returns
        -------
        Tensor
            Transformed data.
        """
        return self.transform(load_tensor(self.path_list[idx]))


@dataclass
class EpisodeDataModuleConfig:
    """EpisodeDataModuleの設定."""

    data_name: str
    batch_size: int
    num_workers: int
    gdrive_url: str
    action_preprocess: Transform
    observation_preprocess: Transform
    action_input_transform: Transform
    action_target_transform: Transform
    observation_input_transform: Transform
    observation_target_transform: Transform

    @property
    def data_dir(self) -> Path:
        """データのディレクトリパス."""
        return Path("data") / self.data_name

    @property
    def processed_data_dir(self) -> Path:
        """加工済みデータのディレクトリパス."""
        return Path("data") / f"{self.data_name}_processed_episode"

    def load_from_gdrive(self) -> None:
        """Google Drive からデータをダウンロードする."""
        filename = gdown.download(self.gdrive_url, quiet=False, fuzzy=True)
        with tarfile.open(filename, "r:gz") as f:
            f.extractall(path=Path("data"), filter="data")
        Path(filename).unlink(missing_ok=False)


class EpisodeDataModule(LightningDataModule):
    """DataModule with actions & observations."""

    def __init__(self, config: EpisodeDataModuleConfig) -> None:
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        """`{data_name}_processed_episode`ディレクトリに加工済みデータを保存する."""
        if not self.config.data_dir.exists():
            self.config.load_from_gdrive()

        if self.config.processed_data_dir.exists():
            return

        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        for action_path in tqdm(sorted(self.config.data_dir.glob("act*"))):
            action = self.config.action_preprocess(load_tensor(action_path))
            new_path = self.config.processed_data_dir / f"{action_path.stem}.pt"
            torch.save(action.detach().clone(), new_path)

        for observation_path in tqdm(sorted(self.config.data_dir.glob("obs*"))):
            observation = self.config.observation_preprocess(load_tensor(observation_path))
            new_path = self.config.processed_data_dir / f"{observation_path.stem}.pt"
            torch.save(observation.detach().clone(), new_path)

    def setup(self, stage: str = "fit") -> None:
        """Create datasets."""
        action_path_list = sorted(self.config.processed_data_dir.glob("act*"))
        observation_path_list = sorted(self.config.processed_data_dir.glob("qua*"))

        train_action_list, val_action_list = split_path_list(action_path_list, 0.8)
        train_observation_list, val_observation_list = split_path_list(observation_path_list, 0.8)

        if stage == "fit":
            self.train_dataset = StackDataset(
                EpisodeDataset(train_action_list, self.config.action_input_transform),
                EpisodeDataset(train_action_list, self.config.action_target_transform),
                EpisodeDataset(train_observation_list, self.config.observation_input_transform),
                EpisodeDataset(train_observation_list, self.config.observation_target_transform),
            )
        self.val_dataset = StackDataset(
            EpisodeDataset(val_action_list, self.config.action_input_transform),
            EpisodeDataset(val_action_list, self.config.action_target_transform),
            EpisodeDataset(val_observation_list, self.config.observation_input_transform),
            EpisodeDataset(val_observation_list, self.config.observation_target_transform),
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
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
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
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )
