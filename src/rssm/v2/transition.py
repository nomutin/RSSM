"""Transition model for RSSM v2."""

import torch
from distribution_extension import MultiDimentionalOneHotCategoricalFactory
from torch import Tensor, nn

from rssm.base.module import Transition
from rssm.base.state import State
from rssm.networks.fc import MLP


class TransitionV2(Transition):
    """
    RSSM V2 Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

    def __init__(
        self,
        deterministic_size: int,
        hidden_size: int,
        action_size: int,
        class_size: int,
        category_size: int,
        activation_name: str,
    ) -> None:
        """Set components."""
        super().__init__()

        self.rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=deterministic_size,
        )
        self.action_state_projector = MLP(
            input_size=action_size + class_size * category_size,
            output_size=hidden_size,
            hidden_size=hidden_size,
            activation_name=activation_name,
            num_hidden_layers=0,
            out_activation_name="Identity",
        )
        self.rnn_to_prior_projector = MLP(
            input_size=deterministic_size,
            output_size=class_size * category_size,
            hidden_size=hidden_size,
            activation_name=activation_name,
            num_hidden_layers=0,
            out_activation_name="Identity",
        )
        self.distribution_factory = MultiDimentionalOneHotCategoricalFactory(
            class_size=class_size,
            category_size=category_size,
        )

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
