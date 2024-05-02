"""Abstract class for RSSM V1/V2 State."""

from collections.abc import Generator

import torch
from distribution_extension import Distribution
from torch import Tensor

Slice = slice | int | tuple[slice | int, ...]


class State:
    """Abstract class for RSSM State."""

    def __init__(
        self,
        deter: Tensor,
        distribution: Distribution,
        stoch: Tensor | None = None,
    ) -> None:
        """Set parameters."""
        self.deter = deter
        self.distribution = distribution
        self.stoch = stoch or distribution.rsample()
        self.feature = torch.cat([self.deter, self.stoch], dim=-1)

    def __iter__(self) -> Generator["State", None, None]:
        """Enable iteration over the batch dimension."""
        i = 0
        while i < self.deter.shape[0]:
            yield self[i]
            i += 1

    def __getitem__(self, loc: Slice) -> "State":
        """Slice the state."""
        return type(self)(
            deter=self.deter[loc],
            stoch=self.stoch[loc],
            distribution=self.distribution[loc],
        )

    def to(self, device: torch.device) -> "State":
        """Move the state to the given device."""
        return type(self)(
            deter=self.deter.to(device),
            stoch=self.stoch.to(device),
            distribution=self.distribution.to(device),
        )

    def detach(self) -> "State":
        """Detach the state."""
        return type(self)(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            distribution=self.distribution.detach(),
        )


def stack_states(states: list[State], dim: int) -> State:
    """Stack states along the given dimension."""
    deter = torch.stack([s.deter for s in states], dim=dim)
    stoch = torch.stack([s.stoch for s in states], dim=dim)
    parameters = {}
    for parameter_name in states[0].distribution.parameters:
        params = [getattr(s.distribution, parameter_name) for s in states]
        parameters[parameter_name] = torch.stack(params, dim=dim)
    distribution = states[0].distribution.__class__(**parameters)
    return State(
        deter=deter,
        stoch=stoch,
        distribution=distribution,
    )


def cat_states(states: list[State], dim: int) -> State:
    """Concatenate states along the given dimension."""
    deter = torch.cat([s.deter for s in states], dim=dim)
    stoch = torch.cat([s.stoch for s in states], dim=dim)
    parameters = {}
    for parameter_name in states[0].distribution.parameters:
        params = [getattr(s.distribution, parameter_name) for s in states]
        parameters[parameter_name] = torch.cat(params, dim=dim)
    distribution = states[0].distribution.__class__(**parameters)
    return State(
        deter=deter,
        stoch=stoch,
        distribution=distribution,
    )
