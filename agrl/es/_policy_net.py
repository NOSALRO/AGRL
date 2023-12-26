import torch
from ._utils import _create_hidden

class PolicyNet(torch.nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layers = None,
            output_activation = None,
        ):
        super().__init__()
        layers_size, activations, other_layers = _create_hidden(hidden_layers)
        layers_size.insert(0, input_dim)
        layers_size.append(output_dim)

        if output_activation is not None:
            activations.append(output_activation)

        # Create MLP.
        self.layers = torch.nn.ModuleList()
        for idx in range(1, len(layers_size)):
            self.layers.append(torch.nn.Linear(layers_size[idx-1] , layers_size[idx]))
            if idx-1 != len(activations):
                self.layers.append(activations[idx-1])
        for layer in other_layers:
            self.layers.insert(layer[0], layer[1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x