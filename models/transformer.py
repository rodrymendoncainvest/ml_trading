import torch
import torch.nn as nn
import math
from core.device import device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


class TransformerModel(nn.Module):
    """
    Transformer Encoder industrial.
    """

    def __init__(self, input_dim, horizon, d_model=64, n_heads=4, num_layers=3, ff_dim=128, dropout=0.1):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.regressor = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos(x)
        out = self.encoder(x)
        last = out[:, -1, :]
        return self.regressor(last)
