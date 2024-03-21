import itertools
import random
from typing import Optional, Tuple, Literal, List, Union, Dict

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from ravt.core.constants import (
    BatchTDict, LossDict, SubsetLiteral
)
from ravt.core.base_classes import BaseSystem, BaseDataSampler, BaseMetric, BaseTransform, BaseDataSource, BaseSAPStrategy
from ravt.core.utils.visualization import draw_feature, draw_bbox, add_clip_id, draw_feature_batch, draw_image, draw_flow
from ravt.core.utils.collection_operations import tensor2ndarray, reverse_collate, select_collate
from ravt.core.utils.array_operations import clip_or_pad_along
from ravt.core.utils.lightning_logger import ravt_logger as logger

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer, concat_pyramids
from .blocks.backbones import YOLOXPAFPNBackbone, DAMOBackbone
from .blocks.necks import LinearTANeck, FPCANeck
from .blocks.heads import TALHead
from .blocks.schedulers import StreamYOLOScheduler, MSCAScheduler
from ravt.data_samplers import YOLOXDataSampler
from ravt.metrics import COCOEvalMAPMetric
from ...core.utils.grad_check import plot_grad_flow


MAX_PAST = 3


class LinearTASystem(YOLOXBaseSystem):
    def __init__(
            self,
            data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
            strategy: Optional[BaseSAPStrategy] = None,
            batch_size: int = 1,
            num_workers: int = 0,

            # structural parameters
            base_depth: int = 3,
            base_channel: int = 64,
            base_neck_depth: int = 3,
            hidden_ratio: float = 1.0,
            strides: Tuple[int, ...] = (8, 16, 32),
            in_channels: Tuple[int, ...] = (256, 512, 1024),
            mid_channel: int = 256,
            depthwise: bool = False,
            act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
            num_classes: int = 8,
            max_objs: int = 100,

            # backbone type
            backbone: Literal['pafpn', 'drfpn'] = 'pafpn',

            # train type
            train_scheduler: Literal['yolox', 'msca'] = 'yolox',

            # predict parameters
            past_time_constant: List[int] = None,
            future_time_constant: List[int] = None,

            # postprocess parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,

            # learning rate parameters
            lr: float = 0.001,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,

            **kwargs,
    ):
        self.save_hyperparameters(ignore=['kwargs', 'data_sources', 'strategy'])

        train_image_clip = [[*past_time_constant, 0]]
        train_bbox_clip = [[0, *future_time_constant]]
        if len(past_time_constant) > 100000000:
            train_image_clip = [[*i, 0] for i in itertools.combinations(past_time_constant, len(past_time_constant)-1)]
            train_bbox_clip = [[0, *future_time_constant]]*len(train_image_clip)
        print(train_image_clip)
        print(train_bbox_clip)

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams) if backbone == 'pafpn' else DAMOBackbone(**self.hparams),
            neck=FPCANeck(**self.hparams),
            head=TALHead(**self.hparams),
            batch_size=batch_size,
            num_workers=num_workers,
            with_bbox_0_train=True,
            data_sources=data_sources,
            data_sampler=YOLOXDataSampler(
                1, [*past_time_constant, 0], future_time_constant,
                train_image_clip, train_bbox_clip
            ),
            metric=COCOEvalMAPMetric(future_time_constant=future_time_constant),
            strategy=strategy,
        )

    def pth_adapter(self, state_dict: Dict) -> Dict:
        if self.hparams.backbone == 'pafpn':
            return super().pth_adapter(state_dict)
        else:
            new_ckpt = {}
            for k, v in state_dict['model'].items():
                k = k.replace('.lateral_conv0.', '.down_conv_5_4.')
                k = k.replace('.C3_p4.', '.down_csp_4_4.')
                k = k.replace('.reduce_conv1.', '.down_conv_4_3.')
                k = k.replace('.C3_p3.', '.down_csp_3_3.')
                k = k.replace('.bu_conv2.', '.up_conv_3_3.')
                k = k.replace('.C3_n3.', '.up_csp_3_4.')
                k = k.replace('.bu_conv1.', '.up_conv_4_4.')
                k = k.replace('.C3_n4.', '.up_csp_4_5.')
                k = k.replace('neck.', 'backbone.neck.')
                new_ckpt[k] = v
            return new_ckpt

    def inference_impl(
            self,
            batch: BatchTDict,
            buffer: Optional[YOLOXBuffer],
    ) -> Tuple[BatchTDict, Optional[Dict]]:
        # TODO: change
        images: Float[torch.Tensor, 'B TP0 C H W'] = batch['image']['image'].float()
        past_time_constant = batch['image']['clip_id'][:, :-1].float()
        future_time_constant = batch['bbox']['clip_id'].float()

        features = self.backbone(images)
        future_features = self.neck.forward(features, past_time_constant, future_time_constant)

        if buffer is not None:
            past_indices = buffer['prev_indices']
            past_features = buffer['prev_features']
        else:
            past_indices = []
            past_features = []

        if past_time_constant is None:
            if len(past_features) > 0:
                ptc = [-1]
                pf = past_features[-1:]
            else:
                ptc = []
                pf = []
            # ptc = [p-default_interval for p in past_indices[-default_max_past:]]

        else:
            interval = past_time_constant[-1]
            ptc = [p-interval for p in past_indices if p-interval in past_time_constant]
            pf = [past_features[i] for i, p in enumerate(past_indices) if p-interval in past_time_constant]

        if future_time_constant is None:
            ftc = [1]
        else:
            ftc = future_time_constant

        # Buffer
        new_buffer = {
            'prev_indices': ptc + [0],
            'prev_features': pf + [features],
        }

        if len(ptc) > 0:
            features = self.neck(
                concat_pyramids(pf + [features]),
                past_time_constant=torch.tensor([ptc], dtype=torch.float32, device=self.device),
                future_time_constant=torch.tensor([ftc], dtype=torch.float32, device=self.device),
            )
        else:
            features = concat_pyramids([features]*len(ftc))
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=50, pad_value=0.0), new_buffer

    def configure_optimizers(self):
        p_backbone_normal, p_backbone_wd = [], []
        p_neck_normal, p_neck_wd = [], []
        p_head_normal, p_head_wd = [], []
        for module, pn, pw in zip(
            [self.backbone, self.neck, self.head],
            [p_backbone_normal, p_neck_normal, p_head_normal],
            [p_backbone_wd, p_neck_wd, p_head_wd],
        ):
            for k, v in module.named_modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                    pn.append(v.bias)
                if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)) or 'bn' in k:
                    pn.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pw.append(v.weight)  # apply decay
        optimizer = torch.optim.SGD(
            [
                {'params': p_backbone_normal},
                {'params': p_backbone_wd, 'weight_decay': self.hparams.weight_decay},
                {'params': p_neck_normal, 'lr': 10*self.hparams.lr},
                {'params': p_neck_wd, 'lr': 10*self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
                {'params': p_head_normal, 'lr': 10*self.hparams.lr},
                {'params': p_head_wd, 'lr': 10*self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            ],
            lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True
        )

        if self.hparams.train_scheduler == 'yolox':
            scheduler = StreamYOLOScheduler(
                optimizer,
                int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
            )
        elif self.hparams.train_scheduler == 'msca':
            scheduler = MSCAScheduler(
                optimizer,
                int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
            )
        else:
            raise ValueError(f'scheduler {self.hparams.scheduler} not supported!')

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]


def linear_ta_s(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        past_time_constant: List[int] = None,
        future_time_constant: List[int] = None,
        num_classes: int = 8,
        base_depth: int = 1,
        base_channel: int = 32,
        base_neck_depth: int = 3,
        hidden_ratio: float = 0.75,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (128, 256, 512),
        mid_channel: int = 128,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> LinearTASystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return LinearTASystem(**__d)


def linear_ta_m(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        past_time_constant: List[int] = None,
        future_time_constant: List[int] = None,
        num_classes: int = 8,
        base_depth: int = 2,
        base_channel: int = 48,
        base_neck_depth: int = 3,
        hidden_ratio: float = 1.0,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (192, 384, 768),
        mid_channel: int = 192,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.4,
        ignore_value: float = 1.7,
        **kwargs
) -> LinearTASystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return LinearTASystem(**__d)


def linear_ta_l(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        past_time_constant: List[int] = None,
        future_time_constant: List[int] = None,
        num_classes: int = 8,
        base_depth: int = 3,
        base_channel: int = 64,
        base_neck_depth: int = 3,
        hidden_ratio: float = 1.0,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (256, 512, 1024),
        mid_channel: int = 256,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.5,
        ignore_value: float = 1.6,
        **kwargs
) -> LinearTASystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return LinearTASystem(**__d)
