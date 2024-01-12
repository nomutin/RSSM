"""Transition model for RSSM v1."""

import torch
from distribution_extention import NormalFactory
from torch import Tensor, nn

from rssm.base.module import Transition, TransitionConfig
from rssm.base.state import State
from rssm.networks.fc import MLP


class TransitionV1(Transition):
    """
    RSSM V1 Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

    def __init__(self, config: TransitionConfig) -> None:
        """Set components."""
        super().__init__()

        self.rnn_cell = nn.GRUCell(
            input_size=config.hidden_size,
            hidden_size=config.deterministic_size,
        )
        self.action_state_projector = MLP(
            input_size=config.action_size + config.stochastic_size,
            output_size=config.hidden_size,
            hidden_size=config.hidden_size,
            activation_name=config.activation_name,
            num_hidden_layers=0,
            out_activation_name="Identity",
        )
        self.rnn_to_prior_projector = MLP(
            input_size=config.deterministic_size,
            output_size=config.stochastic_size * 2,
            hidden_size=config.hidden_size,
            activation_name=config.activation_name,
            num_hidden_layers=0,
            out_activation_name="Identity",
        )
        self.distribution_factory = NormalFactory()

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
        projector_input = torch.cat([action, prev_state.stoch], dim=-1)
        action_state = self.action_state_projector(projector_input)
        deter = self.rnn_cell.forward(action_state, hx=prev_state.deter)
        stoch_source = self.rnn_to_prior_projector(deter)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=deter, distribution=distribution)
