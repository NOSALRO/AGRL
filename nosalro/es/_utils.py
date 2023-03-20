import torch


def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).cpu().detach().numpy()

def set_model_params(model, params, device):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params, device=device), model.parameters())

def _create_hidden(hidden):
    layer_size = []
    activation_funcs = []
    other_layers = []
    for idx, layer in enumerate(hidden):
        if isinstance(layer, tuple):
            layer_size.append(layer[0])
            activation_funcs.append(layer[1])
        else:
            other_layers.append((2*idx, layer))
    return layer_size, activation_funcs, other_layers