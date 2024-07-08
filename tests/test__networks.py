"""Tests of `networks.py`."""

from torch import Tensor

from rssm import State
from rssm.networks import (
    RepresentationV1,
    RepresentationV2,
    TransitionV1,
    TransitionV2,
)
from tests.conftest import (
    ACTION_SIZE,
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    DETERMINISTIC_SIZE,
    HIDDEN_SIZE,
    OBS_EMBED_SIZE,
    STOCHASTIC_SIZE,
)


def test__representation_v1(
    obs_embed_bd: Tensor,
    state_bd: State,
) -> None:
    """Test the RepresentationV1 class and `fowrard()` method."""
    representation = RepresentationV1(
        deterministic_size=DETERMINISTIC_SIZE,
        stochastic_size=STOCHASTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        activation_name="ReLU",
    )
    posterior = representation.forward(
        obs_embed=obs_embed_bd,
        prior_state=state_bd,
    )
    assert posterior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, STOCHASTIC_SIZE)


def test__transition_v1(action_bd: Tensor, state_bd: State) -> None:
    """Test the TransitionV1 class and `fowrard()` method."""
    transition = TransitionV1(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        stochastic_size=STOCHASTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        activation_name="ReLU",
    )
    prior = transition.forward(
        action=action_bd,
        prev_state=state_bd,
    )
    assert prior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, STOCHASTIC_SIZE)


def test__representation_v2(
    obs_embed_bd: Tensor,
    state_discrete_bd: State,
) -> None:
    """Test the RepresentationV2 class and `fowrard()` method."""
    representation = RepresentationV2(
        deterministic_size=DETERMINISTIC_SIZE,
        category_size=CATEGORY_SIZE,
        class_size=CLASS_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        activation_name="ReLU",
    )
    posterior = representation.forward(
        obs_embed=obs_embed_bd,
        prior_state=state_discrete_bd,
    )
    assert posterior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__transition_v2(
    action_bd: Tensor,
    state_discrete_bd: State,
) -> None:
    """Test the TransitionV2 class and `fowrard()` method."""
    transition = TransitionV2(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        category_size=CATEGORY_SIZE,
        class_size=CLASS_SIZE,
        hidden_size=HIDDEN_SIZE,
        activation_name="ReLU",
    )
    prior = transition.forward(
        action=action_bd,
        prev_state=state_discrete_bd,
    )
    assert prior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)
