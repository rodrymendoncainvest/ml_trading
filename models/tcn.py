import torch
import torch.nn as nn
from core.device import device


class Chomp1d(nn.Module):
    """Remove extra padding no fim."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Bloco TCN com:
    - dilated conv
    - residual connection
    - layer norm
    - dropout
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.drop1(out)

        out2 = self.conv2(out)
        out2 = self.chomp2(out2)
        out2 = self.relu2(out2)
        out2 = out2.transpose(1, 2)
        out2 = self.norm2(out2)
        out2 = out2.transpose(1, 2)
        out2 = self.drop2(out2)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out2 + res)


class TCNModel(nn.Module):
    """
    TCN industrial para forecasting.
    Input:  (B, window, features)
    Output: (B, horizon)
    """

    def __init__(self, input_dim, horizon, window,
                 channels=[64, 64, 64], kernel=3, dropout=0.1):

        super().__init__()
        layers = []
        in_ch = input_dim

        dil = 1
        for out_ch in channels:
            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel,
                    dilation=dil,
                    dropout=dropout
                )
            )
            in_ch = out_ch
            dil *= 2

        self.network = nn.Sequential(*layers)

        self.regressor = nn.Linear(in_ch * window, horizon)

    def forward(self, x):
        x = x.transpose(1, 2)  # → (B, features, window)
        out = self.network(x)
        out = out.reshape(out.size(0), -1)
        return self.regressor(out)
