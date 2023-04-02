from typing import Callable
from gym import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class ActorCriticNet(torch.nn.Module):

    def __init__(
            self,
            input_dims,
            output_dims = 2,
        ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.latent_dim_pi = 2
        self.latent_dim_vf = 2
        self.actor_net_create()
        self.critic_net_create()

    def actor_net_create(self):
        self.actor_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dims, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, self.output_dims), torch.nn.Tanh()
        )

    def critic_net_create(self):
        self.critic_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dims, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, self.output_dims), torch.nn.Tanh()
        )

    def actor_forward(self, x):
        return self.actor_net(x)

    def critic_forward(self, x):
        return self.critic_net(x)

    def forward(self, x):
        return self.actor_forward(x), self.critic_forward(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.ortho_init = True

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCriticNet(self.features_dim, 2)
