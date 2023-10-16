from typing import Tuple, List

import torch
import torch.nn as nn
from jaxtyping import Float

from .network_blocks import BaseConv
from .types import PYRAMID


class TANeck(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            past_time_constant: List[int],
            future_time_constant: List[int],
            **kwargs
    ):
        super().__init__()
        ptc = torch.tensor([past_time_constant])
        ftc = torch.tensor([future_time_constant])
        s = torch.exp(ftc.T @ ptc)
        w = s / torch.sum(s, dim=1, keepdim=True) * ftc.T
        self.register_parameter('attn_weight_ta', nn.Parameter(w, requires_grad=True))
        self.convs_f = nn.ModuleList([
            BaseConv(c, c, 1, 1)
            for c in in_channels
        ])

    def forward(self, features: PYRAMID) -> PYRAMID:
        outputs = []
        for i, f in enumerate(features):
            B, TP0, C, H, W = f.size()
            feature_conv = self.convs_f[i](f.flatten(0, 1)).unflatten(0, (B, TP0))
            feature_p = feature_conv[:, -1:] - feature_conv[:, :-1]
            attn = (self.attn_weight_ta @ feature_p.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)

            outputs.append((f[:, -1:] + attn))

        return tuple(outputs)


class TALinearNeck(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            past_time_constant: List[int],
            future_time_constant: List[int],
            **kwargs
    ):
        super().__init__()
        max_time = 10
        ptc = torch.tensor([past_time_constant])
        ftc = torch.tensor([future_time_constant])

        self.register_buffer('c_mask_past', )
        self.fc_temporal = nn.Sequential(
            nn.Linear(max_time, max_time),
            nn.SiLU(),
            nn.Linear(max_time, max_time),
            nn.SiLU(),
        )
        self.convs_f = nn.ModuleList([
            BaseConv(c, c, 1, 1)
            for c in in_channels
        ])

    def forward(self, features: PYRAMID) -> PYRAMID:
        outputs = []
        for i, f in enumerate(features):
            B, TP0, C, H, W = f.size()
            feature_conv = self.convs_f[i](f.flatten(0, 1)).unflatten(0, (B, TP0))
            feature_p = feature_conv[:, -1:] - feature_conv[:, :-1]
            attn = (self.attn_weight_ta @ feature_p.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)

            outputs.append((f[:, -1:] + attn))

        return tuple(outputs)
