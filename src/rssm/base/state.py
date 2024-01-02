"""Abstract class for RSSM V1/V2 State."""


from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Tuple, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from distribution_extention import DistributionBase, Independent

Slice = Union[slice, int, Tuple[Union[slice, int], ...]]


class State:
    """Abstract class for RSSM State."""

    def __init__(self, deter: Tensor, distribution: Independent) -> None:
        """Set parameters."""
        self.deter = deter
        self.distribution = distribution
        self.stoch = distribution.rsample()
        self.feature = torch.cat([self.deter, self.stoch], dim=-1)

    def __iter__(self) -> Generator:
        """Enable iteration over the batch dimension."""
        i = 0
        while i < self.deter.shape[0]:
            yield self[i]
            i += 1

    def __getitem__(self, loc: Slice) -> State:
        """Slice the state."""
        return type(self)(
            deter=self.deter[loc],
            distribution=self.distribution[loc],
        )

    def to(self, device: torch.device) -> State:
        """Move the state to the given device."""
        return type(self)(
            deter=self.deter.to(device),
            distribution=self.distribution.to(device),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the state."""
        return self.deter.shape


def stack_states(states: list[State], dim: int) -> State:
    """Stack states along the given dimension."""
    deter = torch.stack([s.deter for s in states], dim=dim)
    dist: DistributionBase = states[0].distribution.base_dist
    parameter_names = dist.parameters.keys()
    parameters = {}
    for parameter_name in parameter_names:
        params = [
            getattr(state.distribution.base_dist, parameter_name)
            for state in states
        ]
        parameters[parameter_name] = torch.stack(params, dim=dim)
    distribution = dist.__class__(**parameters).independent(dim=1)
    return State(deter=deter, distribution=distribution)


def cat_states(states: list[State], dim: int) -> State:
    """Concatenate states along the given dimension."""
    deter = torch.cat([s.deter for s in states], dim=dim)
    dist: DistributionBase = states[0].distribution.base_dist
    parameter_names = dist.parameters.keys()
    parameters = {}
    for parameter_name in parameter_names:
        params = [
            getattr(state.distribution.base_dist, parameter_name)
            for state in states
        ]
        parameters[parameter_name] = torch.cat(params, dim=dim)
    distribution = dist.__class__(**parameters).independent(dim=1)
    return State(deter=deter, distribution=distribution)
