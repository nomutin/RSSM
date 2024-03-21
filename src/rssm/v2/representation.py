"""Representation model for RSSM V2."""

import torch
from distribution_extension import MultiDimentionalOneHotCategoricalFactory
from torch import Tensor
from torchrl.modules import MLP

from rssm.base.module import Representation
from rssm.base.state import State


class RepresentationV2(Representation):
    """
    Representation model for RSSM V2.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(
        self,
        deterministic_size: int,
        hidden_size: int,
        obs_embed_size: int,
        class_size: int,
        category_size: int,
        activation_name: str,
    ) -> None:
        """Set components."""
        super().__init__()

        self.rnn_to_post_projector = MLP(
            in_features=obs_embed_size + deterministic_size,
            out_features=class_size * category_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.distribution_factory = MultiDimentionalOneHotCategoricalFactory(
            class_size=class_size,
            category_size=category_size,
        )

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
