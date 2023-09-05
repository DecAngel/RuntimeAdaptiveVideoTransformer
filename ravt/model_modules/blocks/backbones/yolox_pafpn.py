from typing import Tuple

import typeguard
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.nn import functional as F

from ..layers import CSPDarknet, DWConv, BaseConv, CSPLayer


class YOLOXPAFPNBackbone(nn.Module):
    """This module extracts the FPN feature of a single image,
    similar to StreamYOLO's DFPPAFPN without the last DFP concat step.

    """
    def __init__(
            self,
            depth_ratio=1.0,
            width_ratio=1.0,
            depthwise=False,
            act='silu',
            **kwargs
    ):
        # select FPN features
        super().__init__()
        feature_base_channels = int(width_ratio * 64)
        feature_channels = tuple(i*feature_base_channels for i in (4, 8, 16))
        feature_depth = int(round(3*depth_ratio))

        self.feature_names = ('dark3', 'dark4', 'dark5')

        # build network (hardcoded)
        self.backbone = CSPDarknet(
            depth_ratio, width_ratio,
            out_features=self.feature_names,
            depthwise=depthwise, act=act
        )
        Conv = DWConv if depthwise else BaseConv

        self.down_conv_5_4 = BaseConv(feature_channels[2], feature_channels[1], 1, 1, act=act)
        self.down_csp_4_4 = CSPLayer(
            2*feature_channels[1],
            feature_channels[1],
            feature_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.down_conv_4_3 = BaseConv(feature_channels[1], feature_channels[0], 1, 1, act=act)
        self.down_csp_3_3 = CSPLayer(
            2*feature_channels[0],
            feature_channels[0],
            feature_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.up_conv_3_3 = Conv(feature_channels[0], feature_channels[0], 3, 2, act=act)
        self.up_csp_3_4 = CSPLayer(
            2*feature_channels[0],
            feature_channels[1],
            feature_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.up_conv_4_4 = Conv(feature_channels[1], feature_channels[1], 3, 2, act=act)
        self.up_csp_4_5 = CSPLayer(
            2*feature_channels[1],
            feature_channels[2],
            feature_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

    @typeguard.typechecked()
    def forward(
            self, image: Float[torch.Tensor, 'batch_size channels_rgb=3 height width'],
    ) -> Tuple[Float[torch.Tensor, 'batch_size channels height width'], ...]:
        """ Extract the FPN feature (p3, p4, p5) of an image tensor of (b, 3, h, w)

        """
        feature = self.backbone(image)
        f3, f4, f5 = list(feature[f_name] for f_name in self.feature_names)

        # 5 -> 4 -> 3 -> 4 -> 5
        x = self.down_conv_5_4(f5)
        m4 = x
        x = F.interpolate(x, size=f4.shape[2:], mode='nearest')
        x = torch.cat([x, f4], dim=1)
        x = self.down_csp_4_4(x)

        x = self.down_conv_4_3(x)
        m3 = x
        x = F.interpolate(x, size=f3.shape[2:], mode='nearest')
        x = torch.cat([x, f3], dim=1)
        x = self.down_csp_3_3(x)
        p3 = x

        x = self.up_conv_3_3(x)
        x = torch.cat([x, m3], dim=1)
        x = self.up_csp_3_4(x)
        p4 = x

        x = self.up_conv_4_4(x)
        x = torch.cat([x, m4], dim=1)
        x = self.up_csp_4_5(x)
        p5 = x

        return p3, p4, p5
