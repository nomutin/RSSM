"""Abstract class for RSSM V1/V2 State."""

from collections.abc import Generator

import torch
from distribution_extension import Distribution
from distribution_extension.utils import cat_distribution, stack_distribution
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
        self.stoch = distribution.rsample() if stoch is None else stoch
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

    def squeeze(self, dim: int) -> "State":
        """Squeeze the state."""
        return type(self)(
            deter=self.deter.squeeze(dim),
            stoch=self.stoch.squeeze(dim),
            distribution=self.distribution.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> "State":
        """Unsqueeze the state."""
        return type(self)(
            deter=self.deter.unsqueeze(dim),
            stoch=self.stoch.unsqueeze(dim),
            distribution=self.distribution.unsqueeze(dim),
        )


def stack_states(states: list[State], dim: int) -> State:
    """Stack states along the given dimension."""
    deter = torch.stack([s.deter for s in states], dim=dim)
    stoch = torch.stack([s.stoch for s in states], dim=dim)
    distribution = stack_distribution([s.distribution for s in states], dim)
    return State(
        deter=deter,
        stoch=stoch,
        distribution=distribution,
    )


def cat_states(states: list[State], dim: int) -> State:
    """Concatenate states along the given dimension."""
    deter = torch.cat([s.deter for s in states], dim=dim)
    stoch = torch.cat([s.stoch for s in states], dim=dim)
    distribution = cat_distribution([s.distribution for s in states], dim)
    return State(
        deter=deter,
        stoch=stoch,
        distribution=distribution,
    )
