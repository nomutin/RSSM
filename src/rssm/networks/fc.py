"""
Modules related to fully-connected layers.

TODO: Remove this
"""

from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron.

    Model Structure
    ---------------
    1. `dense(input_size, hidden_size)`
    2. `activation()`
    3. `dense(hidden_size) + activation()` * `num_hidden_layers`
    4. `dense(hidden_size, output_size)`
    5. `out_activation()`
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        activation_name: str,
        num_hidden_layers: int,
        out_activation_name: str,
    ) -> None:
        """Set parameters and build model."""
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self._activation = getattr(nn, activation_name)
        self._out_activation = getattr(nn, out_activation_name)
        self._model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        seq: list[nn.Module] = [nn.Linear(self.input_size, self.hidden_size)]
        seq += [self._activation()]
        seq += [nn.LayerNorm(self.hidden_size)]
        for _ in range(self.num_hidden_layers):
            seq += self._build_hidden_layer()
        seq += [nn.Linear(self.hidden_size, self.output_size)]
        seq += [self._out_activation()]
        return nn.Sequential(*seq)

    def _build_hidden_layer(self) -> list[nn.Module]:
        layer: list[nn.Module] = []
        layer += [nn.Linear(self.hidden_size, self.hidden_size)]
        layer += [self._activation()]
        layer += [nn.LayerNorm(self.hidden_size)]
        return layer

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass."""
        return self._model(features)
