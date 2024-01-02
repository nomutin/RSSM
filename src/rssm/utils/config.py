"""Utilities for loading and resolving config files."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from pathlib import Path


def add(*x: int) -> int:
    """Add all the arguments."""
    return sum(x)


def mul(*x: int) -> int:
    """Multiply all the arguments."""
    return math.prod(x)


def load_config(path: str | Path) -> DictConfig:
    """Convert model config `.yaml` to `Dictconfig` with custom resolvers."""
    try:
        OmegaConf.register_new_resolver("add", add)
        OmegaConf.register_new_resolver("mul", mul)
    except ValueError:
        pass

    config = OmegaConf.load(path)
    OmegaConf.resolve(config)

    if not isinstance(config, DictConfig):
        msg = "ListConfig does not support"
        raise TypeError(msg)

    return config
