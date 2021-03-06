import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import device


class DQN(nn.Module):
    def __init__(self, layers: tuple):
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(in_features=f1, out_features=f2) if t != 'conv' else
             nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=1) for t, f1, f2 in layers]
        )

    def forward(self, t: torch.tensor):
        t = t.to(device)
        for layer in self.layers:
            t = F.relu(layer(t))
        return t


class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
