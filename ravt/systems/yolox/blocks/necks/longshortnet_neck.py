from typing import Tuple, Optional

import torch
from torch import nn

from ..layers.network_blocks import BaseConv
from ..types import PYRAMID, BaseNeck, TIME


class LongShortNeck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        """Implement LongShortNet with N=3, delta=1, LSFM-Lf-Dil. (The best setting)

        :param in_features: The channels of FPN features.
        """
        super().__init__()
        self.short_convs = nn.ModuleList([
            BaseConv(f, f // 2, ksize=1, stride=1)
            for f in in_channels
        ])
        self.long_convs_1 = nn.ModuleList([
            BaseConv(f, f // 6, ksize=1, stride=1)
            for f in in_channels
        ])
        self.long_convs_2 = nn.ModuleList([
            BaseConv(f, f // 6, ksize=1, stride=1)
            for f in in_channels
        ])
        self.long_convs_3 = nn.ModuleList([
            BaseConv(f, f - (f // 2) - 2*(f // 6), ksize=1, stride=1)
            for f in in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> PYRAMID:
        B, T, _, _, _ = features[0].size()
        assert T == 4
        outputs = []
        for i, f in enumerate(features):
            l3, l2, l1, s1 = f.unbind(1)
            s1 = self.short_convs[i](s1)
            l1 = self.long_convs_1[i](l1)
            l2 = self.long_convs_2[i](l2)
            l3 = self.long_convs_3[i](l3)
            outputs.append((torch.cat([s1, l1, l2, l3], dim=1) + f[:, -1]).unsqueeze(1))

        return tuple(outputs)
