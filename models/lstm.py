import torch
import torch.nn as nn
from core.device import device


class LSTMModel(nn.Module):
    """
    LSTM industrial:
    - 2 camadas
    - hidden 128
    """

    def __init__(self, input_dim, horizon, hidden=128, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.regressor = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.regressor(last)
