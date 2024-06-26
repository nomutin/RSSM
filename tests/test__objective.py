"""Tests of `objective.py`."""

from torch import Tensor

from rssm.objective import likelihood


def test__likelihood(observation_bld: Tensor) -> None:
    """Test the `likelihood()` function."""
    prediction = target = observation_bld
    loss = likelihood(prediction, target, event_ndims=3)
    assert loss.shape == ()
