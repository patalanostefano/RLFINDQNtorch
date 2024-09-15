import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, num_classes, state_size):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 180),
            nn.BatchNorm1d(180),
            nn.Linear(180, 360),
            nn.BatchNorm1d(360),
            nn.Linear(360, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
