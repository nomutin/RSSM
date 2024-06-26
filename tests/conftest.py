"""Common constants and fixtures in the test."""

import pytest
import torch
from distribution_extension import MultiOneHot, Normal
from torch import Tensor

from rssm.state import State

BATCH_SIZE = 4
SEQ_LEN = 8
DETERMINISTIC_SIZE = 64
STOCHASTIC_SIZE = 16
CATEGORY_SIZE = 4
CLASS_SIZE = 4
ACTION_SIZE = 15
HIDDEN_SIZE = 32
OBS_EMBED_SIZE = 7


@pytest.fixture()
def action_bd() -> Tensor:
    """Create a batch of actions."""
    return torch.rand(BATCH_SIZE, ACTION_SIZE)


@pytest.fixture()
def observation_bd() -> Tensor:
    """Create a batch of observations."""
    return torch.rand(BATCH_SIZE, 3, 64, 64)


@pytest.fixture()
def obs_embed_bd() -> Tensor:
    """Create a batch of observation embeddings."""
    return torch.rand(BATCH_SIZE, OBS_EMBED_SIZE)


@pytest.fixture()
def state_bd() -> State:
    """Create a batch of states(continuous)."""
    deter = torch.rand(BATCH_SIZE, DETERMINISTIC_SIZE)
    mean = torch.rand(BATCH_SIZE, STOCHASTIC_SIZE)
    std = torch.rand(BATCH_SIZE, STOCHASTIC_SIZE)
    distribution = Normal(mean, std)
    return State(deter=deter, distribution=distribution)


@pytest.fixture()
def state_discrete_bd() -> State:
    """Create a batch of states(discrete)."""
    deter = torch.rand(BATCH_SIZE, DETERMINISTIC_SIZE)
    logit = torch.rand(BATCH_SIZE, CATEGORY_SIZE, CLASS_SIZE)
    distribution = MultiOneHot(logit)
    return State(deter=deter, distribution=distribution)


@pytest.fixture()
def action_bld() -> Tensor:
    """Create a batch of actions."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, ACTION_SIZE)


@pytest.fixture()
def observation_bld() -> Tensor:
    """Create a batch of observations."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)
