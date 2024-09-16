import torch.nn as nn
import torch.nn.functional as F
from src.agents.dql.dqn import DQN


class RNN_DQN(nn.Module):

    def __init__(self, state_length, action_length, num_layers=1):
        super(RNN_DQN, self).__init__()
        self.lstm = nn.RNN(state_length, hidden_size=60,
                            num_layers=num_layers)
        self.dqn = DQN(60, action_length)

    def forward(self, x):
        # Assuming x is a tensor of shape (batch_size, state_length)

        # Reshape the input to (1, batch_size, state_length) for the RNN
        x = x.unsqueeze(0)

        # Pass the input through the RNN
        output, _ = self.lstm(x)

        # Take the output from the last time step
        output = output[-1, :, :]

        # Pass the output through the DQN
        q_values = self.dqn(output)

        return q_values
