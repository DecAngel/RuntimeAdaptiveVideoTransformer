from typing import List, Tuple

import torch
from torch import nn

from .network_blocks import BaseConv


class DFPMIN(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            width_ratio: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            BaseConv(int(f*width_ratio), int(f*width_ratio), ksize=1, stride=1)
            for f in in_channels
        ])

    def forward(self, features: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> List[torch.Tensor]:
        f0, f1 = features
        f2 = []
        for i in range(len(f0)):
            f2.append(f1[i] + self.convs[i](f1[i]) - self.convs[i](f0[i]))
        return f2
