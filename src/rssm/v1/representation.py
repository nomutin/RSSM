"""Representation model for RSSM V1."""

import torch
from distribution_extension import NormalFactory
from torch import Tensor

from rssm.base.module import Representation
from rssm.base.state import State
from rssm.networks.fc import MLP


class RepresentationV1(Representation):
    """
    Representation model for RSSM V1.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        obs_embed_size: int,
        activation_name: str,
    ) -> None:
        """Set components."""
        super().__init__()

        self.rnn_to_post_projector = MLP(
            input_size=obs_embed_size + deterministic_size,
            output_size=stochastic_size * 2,
            hidden_size=hidden_size,
            activation_name=activation_name,
            num_hidden_layers=0,
            out_activation_name="Identity",
        )
        self.distribution_factory = NormalFactory()

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """
        Single step transition, includes prior transition.

        Parameters
        ----------
        obs_embed : Tensor
            Embedding of observation. Shape: (batch_size, obs_embed_size)
        prior_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Approximate posterior state.

        """
        projector_input = torch.cat([prior_state.deter, obs_embed], -1)
        stoch_source = self.rnn_to_post_projector.forward(projector_input)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=prior_state.deter, distribution=distribution)
