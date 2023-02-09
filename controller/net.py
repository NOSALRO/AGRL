import torch

class Net(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=[256, 128], action_range=1):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_size[0])
        self.l2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        #self.l3 = torch.nn.Linear(hidden_size[1], hidden_size[2])
        self.l4 = torch.nn.Linear(hidden_size[1], output_dim)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        #x = torch.tanh(self.l3(x))
        return self.l4(x)
