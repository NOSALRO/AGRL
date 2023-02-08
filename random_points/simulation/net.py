import torch


class Net(torch.nn.Module):

    def __init__(self, state_dims, action_dims, hidden_sizes = [64]):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dims, hidden_sizes[0])
        self.l2 = torch.nn.Linear(hidden_sizes[0], action_dims)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        return self.l2(x)