"""Loss functions."""

import torch.distributions as td
from torch import Tensor


def likelihood(prediction: Tensor, target: Tensor, event_ndims: int, scale: float = 1.0) -> Tensor:
    """
    Compute the negative log-likelihood.

    Parameters
    ----------
    prediction : Tensor
        Prediction tensor. Shape: [*].
    target : Tensor
        Target tensor. The same shape as `prediction`.
    event_ndims : int
        Number of event dimensions.
    scale : float, optional
        Scale of the distribution, by default 1.0.

    Returns
    -------
    Tensor
        Negative log-likelihood. Shape: [].
    """
    dist = td.Independent(td.Normal(prediction, scale), event_ndims)  # type: ignore[no-untyped-call]
    log_prob: Tensor = dist.log_prob(target)  # type: ignore[no-untyped-call]
    return -log_prob.mean()
