from typing import Tuple, List

import torch
import torch.nn as nn
import math

from jaxtyping import Float
from timm.layers import DropPath
from torch.nn.modules.utils import _pair as to_2tuple
from positional_encodings.torch_encodings import PositionalEncoding1D

from ravt.models.yolox.blocks.layers.network_blocks import BaseConv
from ravt.models.yolox.blocks.types import PYRAMID, BaseNeck


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2,
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            nn.SyncBatchNorm(out_channels // 2),
            nn.SiLU(),
            nn.Conv2d(
                out_channels // 2, out_channels,
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            nn.SyncBatchNorm(out_channels // 2),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.SiLU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.SiLU
                 ):
        super().__init__()
        self.norm1 = nn.SyncBatchNorm(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.SyncBatchNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.SyncBatchNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MSCAN(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4):
        super(MSCAN, self).__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                )

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         )
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    nn.init.trunc_normal_(p, std=0.02, b=0.0)
            elif isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    nn.init.constant_(p, val=1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                for p in m.parameters():
                    nn.init.normal_(p, mean=0, std=math.sqrt(2.0 / fan_out))

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MSTCASTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.temporal_encoding = PositionalEncoding1D(dim // 4)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.SiLU()
        self.conv_0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.max_pool_temporal = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool_temporal = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_temporal = nn.MultiheadAttention(dim // 4, num_heads=1, batch_first=True)
        self.linear_temporal_1 = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.SiLU(),
        )
        self.linear_temporal_2 = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
        )
        self.conv_1 = nn.Conv2d(dim, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(
            self, x: Float[torch.Tensor, 'batch_size time channel height width']
    ) -> Float[torch.Tensor, 'batch_size channel height width']:
        B, T, C, H, W = x.size()
        x_current = x[:, -1]

        feature = x.reshape(B*T, C, H, W)
        feature = self.proj_1(feature)
        feature = self.activation(feature)

        feature_1 = self.conv_0(feature)
        feature_1 = self.conv_spatial(feature_1)
        feature_pooled = torch.cat([self.max_pool_temporal(feature_1), self.avg_pool_temporal(feature_1)], dim=1)
        feature_pooled = feature_pooled.reshape(B, T, 2*C)
        feature_pooled = self.linear_temporal_1(feature_pooled)
        feature_pooled = feature_pooled + self.temporal_encoding(feature_pooled)
        feature_pooled, attn_weight = self.attn_temporal(feature_pooled, feature_pooled, feature_pooled)
        feature_pooled = self.linear_temporal_2(feature_pooled)

        feature_2 = feature_1.reshape(B, T, C, H, W)[:, -1] + feature_pooled[:, -1, :, None, None]
        feature_2 = self.conv_1(feature_2)

        feature = feature.reshape(B, T, C, H, W)[:, -1]*feature_2
        feature = self.proj_2(feature)

        return x_current + feature


class MSTCABlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.SiLU):
        super().__init__()
        self.norm1 = nn.SyncBatchNorm(dim)
        self.attn = MSTCASTAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.SyncBatchNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        layer_scale_init_value = torch.tensor(1e-2)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True
        )

    def forward(
            self, x: Tuple[Float[torch.Tensor, 'batch_size time channel height width']]
    ) -> Tuple[Float[torch.Tensor, 'batch_size channel height width']]:
        x_t = x[:, -1]
        x_t = x_t + self.drop_path(
            self.layer_scale_1 * self.attn(torch.stack([self.norm1(xi) for xi in x.unbind(1)], dim=1))
        )
        x_t = x_t + self.drop_path(
            self.layer_scale_2 * self.mlp(self.norm2(x_t))
        )
        return x_t


class MSCANeck(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...], **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([MSTCASTAttention(c) for c in in_channels])

    def forward(self, features: Tuple[Float[torch.Tensor, 'batch_size time channels height width'], ...]):
        outputs = []
        for f, b in zip(features, self.blocks):
            outputs.append(b(f))
        return tuple(outputs)


class SimpleNeck(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...], time_constant: List[int], **kwargs):
        super().__init__()
        t = len(time_constant) + 1
        self.convs_1 = nn.ModuleList([
            BaseConv(c, c, 1, 1)
            for c in in_channels
        ])
        self.linears_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(t * c, (t-1) * c),
                nn.SiLU(),
            )
            for c in in_channels
        ])
        self.p_tc = nn.Parameter(
            torch.tensor(time_constant, dtype=torch.float32).reshape(1, len(time_constant), 1, 1, 1),
            requires_grad=True,
        )

    def forward(self, features: Tuple[Float[torch.Tensor, 'batch_size time channels height width'], ...]):
        outputs = []

        for f, conv_1, linear_1 in zip(features, self.convs_1, self.linears_1):
            B, T, C, H, W = f.size()
            bt_c_h_w = f.reshape(B*T, C, H, W)
            bt_c_h_w = conv_1(bt_c_h_w)

            b_h_w_ct = bt_c_h_w.reshape(B, T, C, H, W).permute(0, 3, 4, 2, 1).flatten(3, 4)
            b_h_w_ct1 = linear_1(b_h_w_ct)

            b_t_c_h_w = bt_c_h_w.reshape(B, T, C, H, W)
            b_t1_c_h_w = b_t_c_h_w[:, -1:] - b_t_c_h_w[:, :-1]
            b_t1_c_h_w = b_t1_c_h_w * self.p_tc

            b_t1_c_h_w = b_t1_c_h_w * b_h_w_ct1.reshape(B, H, W, C, T-1).permute(0, 4, 3, 1, 2)
            b_c_h_w = torch.sum(b_t1_c_h_w, dim=1)

            outputs.append(f[:, -1] + b_c_h_w)
        return tuple(outputs)


class Simple2Block(nn.Module):
    def __init__(
            self,
            in_channel: int,
            # hidden_channel: int,
    ):
        super().__init__()
        self.fc_time_constant = nn.Sequential(
            nn.Linear(1, in_channel),
            nn.SiLU(),
            nn.LayerNorm(in_channel),
        )
        self.fc_in = nn.Sequential(
            nn.Linear(2 * in_channel, in_channel),
            nn.SiLU(),
        )
        self.conv_p = BaseConv(in_channel, in_channel, 1, 1)
        self.spatial_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm_attn = nn.LayerNorm(in_channel)
        self.norm_mlp = nn.LayerNorm(in_channel)
        self.fc_query = nn.Linear(in_channel, in_channel)
        self.fc_key = nn.Linear(in_channel, in_channel)
        self.fc_value = nn.Linear(in_channel, in_channel)
        self.fc_mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.SiLU(),
        )
        self.p_attn = nn.Parameter(
            torch.tensor(1.0).repeat(1, 1, in_channel, 1, 1), requires_grad=True
        )
        self.p_mlp = nn.Parameter(
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
        ptc = self.fc_time_constant(past_time_constant[:, None])[None, :, :]        # 1 TP C
        ftc = self.fc_time_constant(future_time_constant[:, None])[None, :, :]      # 1 TF C

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
        # B TF H W C
        attn = attn.unflatten(2, (C, H, W)).permute(0, 1, 3, 4, 2)
        # B TF C H W
        attn = self.norm_attn(attn).permute(0, 1, 4, 2, 3)
        # B TF C H W
        feature_f = feature_f + attn * self.p_attn

        # B TF H W C
        mlp = self.fc_mlp(feature_f.flatten(0, 1)).unflatten(0, (B, TP)).permute(0, 1, 3, 4, 2)
        # B TF C H W
        feature_f = feature_f + self.norm_mlp(mlp).permute(0, 1, 4, 2, 3) * self.p_mlp

        return feature_f

    def forward2(
            self,
            feature: Float[torch.Tensor, 'batch_size time_p channels height width'],
            past_time_constant: Float[torch.Tensor, 'time_p'],
            future_time_constant: Float[torch.Tensor, 'time_f'],
    ) -> Float[torch.Tensor, 'batch_size time_f channels height width']:
        B, TP, CI, H, W = feature.size()
        ptc = self.fc_time_constant(past_time_constant[:, None])[None, :, :]        # 1 TP CH
        ftc = self.fc_time_constant(future_time_constant[:, None])[None, :, :]      # 1 TF CH
        feature = feature.reshape(B, H, W, TP, CI).flatten(0, 2)                    # BHW TP CI

        feature_now = feature[:, -1]                                                # BHW CI
        feature_past = self.fc_in(feature + ptc)                                    # BHW TP CH
        feature_future = self.fc_in(feature_now + ftc)                              # BHW TF CH

        query = self.fc_query(feature_future)                                       # BHW TF C
        key = self.fc_key(feature_past)                                             # BHW TP C
        value = self.fc_value(feature_past)                                         # BHW TP C

        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn = attn_weight @ value                                                  # BHW TF C
        attn = self.norm(attn)                                                      # BHW TF C
        ff = feature_now + attn                                                     # BHW TF C

        ff = ff + self.fc_mlp(ff)                                                   # BHW TF C
        return ff.unflatten(0, (B, H, W)).permute(0, 3, 4, 1, 2).contiguous()


class Simple2Neck(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            past_time_constant: List[int],
            future_time_constant: List[int],
            **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Simple2Block(c)
            for c in in_channels
        ])
        self.register_buffer('ptc', torch.tensor(past_time_constant, dtype=torch.float32), persistent=False)
        self.register_buffer('ftc', torch.tensor(future_time_constant, dtype=torch.float32), persistent=False)

    def forward(self, features: Tuple[Float[torch.Tensor, 'batch_size time channels height width'], ...]):
        outputs = []
        for f, block in zip(features, self.blocks):
            outputs.append(block(f, self.ptc, self.ftc).squeeze(1))
        return tuple(outputs)


class Simple3Neck(nn.Module):
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

    def forward(self, features: Tuple[Float[torch.Tensor, 'batch_size time channels height width'], ...]):
        outputs = []
        for i, f in enumerate(features):
            B, TP0, C, H, W = f.size()
            feature_conv = self.convs_f[i](f.flatten(0, 1)).unflatten(0, (B, TP0))
            feature_p = feature_conv[:, -1:] - feature_conv[:, :-1]
            attn = (self.attn_weight_ta @ feature_p.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)

            outputs.append((f[:, -1:] + attn).squeeze(1))

        return tuple(outputs)
