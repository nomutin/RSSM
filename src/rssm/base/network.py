"""Abstract classes for RSSM."""

from torch import Tensor, nn

from rssm.state import State


class Representation(nn.Module):
    """
    RSSM Representation Model.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.rnn_to_post_projector = nn.Module()
        self.distribution_factory = nn.Module()

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """Single step transition, includes prior transition."""
        raise NotImplementedError


class Transition(nn.Module):
    """
    RSSM Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.rnn_cell = nn.Module()
        self.action_state_projector = nn.Module()
        self.rnn_to_prior_projector = nn.Module()
        self.distribution_factory = nn.Module()

    def forward(self, action: Tensor, prev_state: State) -> State:
        """
        Single step transition, includes deterministic transitions by GRUs.

        Parameters
        ----------
        action : Tensor
            (Prev) aciton of agent or robot. Shape: (batch_size, action_size)
        prev_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Prior state.

        """
        raise NotImplementedError
