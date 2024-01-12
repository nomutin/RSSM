"""Representation model for RSSM V2."""

import torch
from distribution_extention import MultiDimentionalOneHotCategoricalFactory
from torch import Tensor

from rssm.base.module import Representation, RepresentationConfig
from rssm.base.state import State
from rssm.networks.fc import MLP


class RepresentationV2(Representation):
    """
    Representation model for RSSM V2.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(self, config: RepresentationConfig) -> None:
        """Set components."""
        super().__init__()

        self.rnn_to_post_projector = MLP(
            input_size=config.obs_embed_size + config.deterministic_size,
            output_size=config.stochastic_size,
            hidden_size=config.hidden_size,
            activation_name=config.activation_name,
            num_hidden_layers=0,
            out_activation_name="Identity",
        )
        self.distribution_factory = MultiDimentionalOneHotCategoricalFactory(
            class_size=config.class_size,
            category_size=config.category_size,
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
        stoch_source = self.rnn_to_post_projector.forward(projector_input)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=prior_state.deter, distribution=distribution)
