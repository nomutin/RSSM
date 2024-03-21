"""Representation model for RSSM V1."""

import torch
from distribution_extension import NormalFactory
from torch import Tensor
from torchrl.modules import MLP

from rssm.base.module import Representation
from rssm.base.state import State


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
            in_features=obs_embed_size + deterministic_size,
            out_features=stochastic_size * 2,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
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
        stoch_source = self.rnn_to_post_projector(projector_input)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=prior_state.deter, distribution=distribution)
