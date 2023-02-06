import torch

class Net(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=[256, 128], action_range=1):

        super().__init__()

        self.l1 = torch.nn.Linear(input_dim, hidden_size[0])
        self.l2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = torch.nn.Linear(hidden_size[1], output_dim)
        self.action_range = action_range

    def forward(self, x):

        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        # return torch.tanh(self.l3(x)).view(2,3) * self.action_range
        return torch.tanh(self.l3(x)) * self.action_range
