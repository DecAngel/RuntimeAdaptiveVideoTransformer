from typing import Tuple, Optional, Literal

import timm
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn
from kornia.geometry.transform import remap
from kornia.utils.grid import create_meshgrid
from torch.nn import Conv2d

from torchvision.ops import DeformConv2d, deform_conv2d
from ..layers.network_blocks import BaseConv
from ..types import PYRAMID, BaseNeck, TIME

MAX_H, MAX_W = 1000, 1000


class DeformNeck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        super().__init__()
        # channels and strides
        base_channel = in_channels[0]

        # coordinates (1, H, W, 2)
        self.register_buffer('grid', create_meshgrid(MAX_H, MAX_W, False, dtype=torch.float32), persistent=False)

        self.offset_1_convs = nn.ModuleList([
            BaseConv(4*c, base_channel, ksize=1, stride=1)
            for c in in_channels
        ])
        self.offset_2_conv = Conv2d(len(in_channels)*base_channel, 2, 1, 1)

    def forward(
            self, features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None
    ) -> PYRAMID:
        heights, widths = [], []
        for f in features:
            B, TP, C, H, W = f.size()
            assert TP == 4
            heights.append(H)
            widths.append(W)
        base_h, base_w = heights[-1], widths[-1]

        maps = []
        for f, h, w, conv in zip(features, heights, widths, self.offset_1_convs):
            m = conv(f.flatten(1, 2))
            m = F.interpolate(m, size=(base_h, base_w), mode='bilinear', align_corners=False)
            maps.append(m)

        m = torch.cat(maps, dim=1)
        offset = self.offset_2_conv(m)
        if self.training:
            self.vis_offset = offset

        outputs = []
        for f, h, w in zip(features, heights, widths):
            resize_h, resize_w = h / base_h, w / base_w
            resized_factor = torch.tensor([resize_h, resize_w], dtype=torch.float32, device=f.device)
            resized_offset = F.interpolate(offset, size=(h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            grid = self.grid[:, :h, :w, :] + resized_offset * resized_factor
            output = remap(f[:, -1], grid[..., 0], grid[..., 1], mode='bilinear', align_corners=True)
            outputs.append(output.unsqueeze(1))

        return tuple(outputs)


"""
class TemporalFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.functions = [
            lambda x: torch.ones_like(x),
            lambda x: x,
            lambda x: x ** 2,
            lambda x: x ** 3,
            lambda x: torch.sin(torch.pi * x),
            lambda x: torch.sin(torch.pi * x / 2),
            lambda x: torch.sin(torch.pi * x / 4),
        ]

    @property
    def out_channels(self):
        return len(self.functions)

    def forward(
            self, time_constant: Float[torch.Tensor, 'batch_size time']
    ) -> Float[torch.Tensor, 'batch_size time channels']:
        return torch.stack([f(time_constant) for f in self.functions], dim=-1)

self.tpe_p = nn.Embedding(10, base_channel)
self.tpe_f = nn.Embedding(10, base_channel)
tpe_p = torch.cumprod(F.sigmoid(self.tpe_p(
    torch.arange(0, torch.max(-past_time_constant), device=past_time_constant.device)
)), dim=0)
tpe_p = torch.stack([tpe_p[ptc] for ptc in -past_time_constant])

tpe_f = torch.cumprod(F.sigmoid(self.tpe_f(
    torch.arange(0, torch.max(future_time_constant), device=future_time_constant.device)
)), dim=0)
tpe_f = torch.stack([tpe_f[ftc] for ftc in future_time_constant])
"""
