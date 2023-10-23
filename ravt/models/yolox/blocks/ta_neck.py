import math
from typing import Tuple, List, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

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


class TA2Neck(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            past_time_constant: List[int],
            future_time_constant: List[int],
            **kwargs
    ):
        super().__init__()
        self.register_buffer('ptc', torch.tensor(past_time_constant)[None, :, None])
        self.register_buffer('ftc', torch.tensor(future_time_constant)[None, :, None])
        self.convs_f = nn.ModuleList([
            BaseConv(c, c, 1, 1)
            for c in in_channels
        ])
        self.fc_time = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, c // 2),
                nn.Tanh(),
                nn.Linear(c // 2, c),
                nn.Tanh(),
            )
            for c in in_channels
        ])
        self.fc_query = nn.ModuleList([
            nn.Linear(c, c)
            for c in in_channels
        ])
        self.fc_key = nn.ModuleList([
            nn.Linear(c, c)
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
            feature: Float[torch.Tensor, 'batch_size time_p channels height width'],
            past_time_constant: Float[torch.Tensor, 'time_p'],
            future_time_constant: Float[torch.Tensor, 'time_f'],
    ) -> Float[torch.Tensor, 'batch_size time_f channels height width']:
        B, _, C, H, W = feature.size()
        TP = past_time_constant.size(0)
        TF = future_time_constant.size(0)

        feature_p = self.conv_p(feature.flatten(0, 1)).unflatten(0, (B, TP + 1))    # B TP C H W
        feature_f = feature[:, -1:].expand(-1, TF, -1, -1, -1)                      # B TF C H W
        ptc = self.fc_time_constant(past_time_constant[None, :, None])              # 1 TP C
        ftc = self.fc_time_constant(future_time_constant[None, :, None])            # 1 TF C

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
        attn_value = (feature_p[:, -1:]-feature_p[:, :-1]).flatten(2, 4)                                        # B TP CHW

        # B TF CHW
        attn_weight = attn_query @ attn_key.transpose(-2, -1) / math.sqrt(attn_query.size(-1))
        attn = attn_weight @ attn_value         # BHW TF C
        # attn = nn.functional.scaled_dot_product_attention(attn_query, attn_key, attn_value)
        # B TF C H W
        attn = attn.unflatten(2, (C, H, W))
        # B TF C H W
        feature_f = feature_f + attn * self.p_attn

        return feature_f


class TA5Neck(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            past_time_constant: List[int],
            future_time_constant: List[int],
            **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TA5Block(c)
            for c in in_channels
        ])
        self.register_buffer('ptc', torch.tensor(past_time_constant, dtype=torch.float32), persistent=False)
        self.register_buffer('ftc', torch.tensor(future_time_constant, dtype=torch.float32), persistent=False)

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Union[List[int], Int[torch.Tensor, 'TP'], None] = None,
            future_time_constant: Union[List[int], Int[torch.Tensor, 'TF'], None] = None,
    ) -> PYRAMID:
        outputs = []
        if past_time_constant is None:
            past_time_constant = self.ptc
        elif isinstance(past_time_constant, list):
            past_time_constant = torch.tensor(past_time_constant, dtype=torch.int32, device=self.ptc.device)
        if future_time_constant is None:
            future_time_constant = self.ftc
        elif isinstance(future_time_constant, list):
            future_time_constant = torch.tensor(future_time_constant, dtype=torch.int32, device=self.ftc.device)

        for f, block in zip(features, self.blocks):
            outputs.append(block(f, past_time_constant, future_time_constant))
        return tuple(outputs)
