"""Neural model: the dual-resolution TCNN-LSTM ported from the ADDIM HAR architecture.

The published Human Activity Recognition model combines a short-context branch and a
long-context branch, each a temporal convolutional stack, whose pooled features are fused
and passed through a recurrent step. Here the same structure is transferred to volatility
forecasting: a short branch sees a 5-day window of log-RV and a long branch sees a 22-day
window, mirroring the daily/weekly/monthly horizons of HAR-RV. Each branch is a stack of
dilated 1D convolutions (kernel 3, dilations 1, 2, 4); the two pooled feature vectors are
concatenated and treated as a length-1 sequence fed to a single LSTM step, then an MLP head
emits a scalar log-RV forecast. Input channels are 1 (univariate log-RV) and the loss is MSE,
the two adaptations from the original classification model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TCNBranch(nn.Module):
    """Stack of dilated 1D convolutions followed by global average pooling."""

    def __init__(self, in_ch: int = 1, channels=(32, 64), kernel: int = 3,
                 dilations=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        # One conv per dilation; channel widths cycle [32, 64, 64] for dilations [1, 2, 4].
        widths = [channels[0]] + [channels[1]] * (len(dilations) - 1)
        layers = []
        c_in = in_ch
        for d, c_out in zip(dilations, widths):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel, dilation=d, padding=d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_ch, T)
        h = self.net(x)               # (B, C, T) -- padding=dilation keeps length T
        return h.mean(dim=2)          # global average pool -> (B, C)


class DualResTCNNLSTM(nn.Module):
    """Short (5-day) and long (22-day) TCN branches fused through a single LSTM step."""

    def __init__(self, short_window: int = 5, long_window: int = 22,
                 tcnn_channels=(32, 64), lstm_hidden: int = 64, dropout: float = 0.1,
                 branches: str = "both"):
        super().__init__()
        assert branches in ("both", "short", "long")
        self.branches = branches
        self.short_window = short_window
        self.long_window = long_window
        self.short_branch = TCNBranch(1, tcnn_channels, dropout=dropout)
        self.long_branch = TCNBranch(1, tcnn_channels, dropout=dropout)
        c = self.short_branch.out_channels
        fused = 2 * c if branches == "both" else c
        self.lstm = nn.LSTM(input_size=fused, hidden_size=lstm_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x_short: torch.Tensor, x_long: torch.Tensor) -> torch.Tensor:
        if self.branches == "both":
            feat = torch.cat([self.short_branch(x_short), self.long_branch(x_long)], dim=1)
        elif self.branches == "short":
            feat = self.short_branch(x_short)
        else:
            feat = self.long_branch(x_long)
        out, _ = self.lstm(feat.unsqueeze(1))           # (B, 1, F): sequence of length 1
        return self.head(out[:, -1, :]).squeeze(-1)     # (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
