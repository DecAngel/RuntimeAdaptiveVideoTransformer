from typing import Tuple, Optional

import torch
from torch import nn

from ..layers.network_blocks import BaseConv
from ..types import PYRAMID, BaseNeck, TIME


class DFP(BaseNeck):
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

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> PYRAMID:
        B, T, _, _, _ = features[0].size()

        outputs = []
        for i, conv in enumerate(self.convs):
            features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, T)).flip(1)
            features_conv = torch.cat([features_conv[:, :1], features_conv[:, 1:]], dim=2)
            outputs.append(features[i][:, -1:]+features_conv)

        return tuple(outputs)
