"""Abstract class for RSSM V1/V2 State."""


from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Tuple, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from distribution_extention import Distribution

Slice = Union[slice, int, Tuple[Union[slice, int], ...]]


class State:
    """Abstract class for RSSM State."""

    def __init__(self, deter: Tensor, distribution: Distribution) -> None:
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

    def squeeze(self, dim: int) -> State:
        """Squeeze the state along the given dimension."""
        return type(self)(
            deter=self.deter.squeeze(dim),
            distribution=self.distribution.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> State:
        """Unsqueeze the state along the given dimension."""
        return type(self)(
            deter=self.deter.unsqueeze(dim),
            distribution=self.distribution.unsqueeze(dim),
        )

    def detach(self) -> State:
        """Detach the state."""
        return type(self)(
            deter=self.deter.detach(),
            distribution=self.distribution.detach(),
        )


def stack_states(states: list[State], dim: int) -> State:
    """Stack states along the given dimension."""
    deter = torch.stack([s.deter for s in states], dim=dim)
    dist: Distribution = states[0].distribution

    parameters = {}
    for parameter_name in dist.arg_constraints:
        params = [getattr(s.distribution, parameter_name) for s in states]
        parameters[parameter_name] = torch.stack(params, dim=dim)
    distribution = dist.__class__(**parameters)
    return State(deter=deter, distribution=distribution)


def cat_states(states: list[State], dim: int) -> State:
    """Concatenate states along the given dimension."""
    deter = torch.cat([s.deter for s in states], dim=dim)
    dist: Distribution = states[0].distribution
    parameter_names = dist.parameters.keys()
    parameters = {}
    for parameter_name in parameter_names:
        params = [getattr(s.distribution, parameter_name) for s in states]
        parameters[parameter_name] = torch.cat(params, dim=dim)
    distribution = dist.__class__(**parameters)
    return State(deter=deter, distribution=distribution)
