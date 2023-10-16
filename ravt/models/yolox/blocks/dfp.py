from typing import Tuple

import torch
from torch import nn

from .network_blocks import BaseConv
from .types import PYRAMID


class DFP(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            BaseConv(f, f // 2, ksize=1, stride=1)
            for f in in_channels
        ])

    def forward(self, features: PYRAMID) -> PYRAMID:
        B, T, _, _, _ = features[0].size()
        assert T == 2
        f0, f1 = [[f[:, t] for f in features] for t in range(T)]
        f2 = []
        for i in range(len(f0)):
            f2.append((f1[i] + torch.cat([self.convs[i](f1[i]), self.convs[i](f0[i])], dim=1)).unsqueeze(1))
        return tuple(f2)
