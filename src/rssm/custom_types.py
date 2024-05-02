"""Custom types for RSSM."""

from typing import TypeAlias

from torch import Tensor

DataGroup: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]
LossDict: TypeAlias = dict[str, Tensor]
