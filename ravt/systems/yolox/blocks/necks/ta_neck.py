import math
from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from ..layers.network_blocks import BaseConv
from ..types import PYRAMID, BaseNeck, TIME


class TA5Block(nn.Module):
    def __init__(
            self,
            in_channel: int,
    ):
        super().__init__()
        self.fc_time_constant = nn.Sequential(
            nn.Linear(1, in_channel),
            nn.Tanh(),
        )
        self.fc_in = nn.Sequential(
            nn.Linear(2 * in_channel, in_channel),
            nn.SiLU(),
        )
        self.conv_p = BaseConv(in_channel, in_channel, 1, 1)
        self.spatial_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_query = nn.Linear(in_channel, in_channel)
        self.fc_key = nn.Linear(in_channel, in_channel)
        self.p_attn = nn.Parameter(
            torch.tensor(1.0).repeat(1, 1, in_channel, 1, 1), requires_grad=True
        )

    def forward(
            self,
            feature: Float[torch.Tensor, 'batch_size time_p_1 channels height width'],
            past_time_constant: Float[torch.Tensor, 'batch_size time_p'],
            future_time_constant: Float[torch.Tensor, 'batch_size time_f'],
    ) -> Float[torch.Tensor, 'batch_size time_f channels height width']:
        B, _, C, H, W = feature.size()
        TP = past_time_constant.size(1)
        TF = future_time_constant.size(1)

        feature_p = self.conv_p(feature.flatten(0, 1)).unflatten(0, (B, TP + 1))    # B TP C H W
        feature_f = feature[:, -1:].expand(-1, TF, -1, -1, -1)                      # B TF C H W
        ptc = self.fc_time_constant(past_time_constant[..., None])                  # B TP C
        ftc = self.fc_time_constant(future_time_constant[..., None])                # B TF C

        # B TP C
        attn_in_p = feature_p[:, :-1].flatten(0, 1)
        attn_in_p = torch.cat([self.spatial_max_pool(attn_in_p), self.spatial_avg_pool(attn_in_p)], dim=1)
        attn_in_p = attn_in_p.unflatten(0, (B, TP)).squeeze(-1).squeeze(-1)
        attn_in_p = self.fc_in(attn_in_p) + ptc
        # B TF C
        attn_in_f = feature_f.flatten(0, 1)
        attn_in_f = torch.cat([self.spatial_max_pool(attn_in_f), self.spatial_avg_pool(attn_in_f)], dim=1)
        attn_in_f = attn_in_f.unflatten(0, (B, TF)).squeeze(-1).squeeze(-1)
        attn_in_f = self.fc_in(attn_in_f) + ftc

        attn_query = self.fc_query(attn_in_f)                                       # B TF C
        attn_key = self.fc_key(attn_in_p)                                           # B TP C
        attn_value = (feature_p[:, -1:]-feature_p[:, :-1]).flatten(2, 4)            # B TP CHW

        # B TF CHW
        attn_weight = attn_query @ attn_key.transpose(-2, -1) / math.sqrt(attn_query.size(1))
        attn = attn_weight @ attn_value         # BHW TF C
        # attn = nn.functional.scaled_dot_product_attention(attn_query, attn_key, attn_value)
        # B TF C H W
        attn = attn.unflatten(2, (C, H, W))
        # B TF C H W
        feature_f = feature_f + attn * self.p_attn / attn_key.size(1)

        return feature_f


class TA5Neck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TA5Block(c)
            for c in in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> PYRAMID:
        outputs = []
        if past_time_constant is None:
            TP = features[0].size(1) - 1
            past_time_constant = torch.arange(-TP, -1, step=1, dtype=torch.float32, device=features[0].device)[None, ...]
        if future_time_constant is None:
            future_time_constant = torch.tensor([[1]], dtype=torch.float32, device=features[0].device)

        for f, block in zip(features, self.blocks):
            outputs.append(block(f, past_time_constant, future_time_constant))
        return tuple(outputs)
