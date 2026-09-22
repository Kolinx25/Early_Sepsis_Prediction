import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.norm = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.conv(x)

        # remove future leakage from padding
        x = x[:, :, : -self.conv.padding[0]]

        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class SepsisTCN(nn.Module):
    def __init__(self, input_channels=69):
        super().__init__()

        self.network = nn.Sequential(
            TemporalBlock(input_channels, 64, dilation=1),
            TemporalBlock(64, 64, dilation=2),
            TemporalBlock(64, 128, dilation=4),
        )

        self.classifier = nn.Linear(128, 1)

    def forward(self, x):

        # x:
        # batch, 12, 69

        x = x.transpose(1, 2)

        # batch,69,12

        x = self.network(x)

        # temporal pooling

        x = torch.mean(x, dim=2)

        logits = self.classifier(x)

        return logits.squeeze(1)
