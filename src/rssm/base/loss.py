"""Loss functions."""

import torch.distributions as td
from torch import Tensor, nn


def likelihood(prediction: Tensor, target: Tensor, event_ndims: int, scale: float = 1.0) -> Tensor:
    """Compute the negative log-likelihood."""
    dist = td.Independent(td.Normal(prediction, scale), event_ndims)
    return -dist.log_prob(target).mean()


def mse(prediction: Tensor, target: Tensor, factor: float = 1.0) -> Tensor:
    """Compute mean squared error."""
    return nn.MSELoss()(prediction, target).mean().mul(factor)
