from typing import Optional, Tuple, Literal, List, Dict

import numpy as np
import torch

from ravt.core.base_classes import BaseDataSource, BaseSAPStrategy, BaseDataSampler, BaseTransform, BaseMetric
from ravt.core.constants import SubsetLiteral, BatchTDict
from ravt.core.utils.array_operations import clip_or_pad_along

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer
from .blocks.backbones import YOLOXPAFPNBackbone
from .blocks.necks import IdentityNeck
from .blocks.heads import YOLOXHead
from ravt.data_samplers import YOLOXDataSampler
from ravt.metrics import COCOEvalMAPMetric


class YOLOXSystem(YOLOXBaseSystem):
    def __init__(
            self,
            data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
            strategy: Optional[BaseSAPStrategy] = None,
            batch_size: int = 1,
            num_workers: int = 0,

            # structural parameters
            base_depth: int = 3,
            base_channel: int = 64,
            strides: Tuple[int, ...] = (8, 16, 32),
            in_channels: Tuple[int, ...] = (256, 512, 1024),
            mid_channel: int = 256,
            depthwise: bool = False,
            act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
            num_classes: int = 8,
            max_objs: int = 100,
            gamma: float = 1.0,
            ignore_thr: float = 0.5,
            ignore_value: float = 1.5,

            # predict parameters
            predict_num: int = 0,

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

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams),
            neck=IdentityNeck(**self.hparams),
            head=YOLOXHead(**self.hparams),
            batch_size=batch_size,
            num_workers=num_workers,
            with_bbox_0_train=False,
            data_sources=data_sources,
            data_sampler=YOLOXDataSampler(1, [0], [predict_num], [[0]], [[predict_num]]),
            metric=COCOEvalMAPMetric(future_time_constant=[predict_num]),
            strategy=strategy,
        )

    def inference_impl(
            self,
            batch: BatchTDict,
            buffer: Optional[YOLOXBuffer],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BatchTDict, Optional[Dict]]:
        # TODO: change
        # Ignore buffer, ptc and ftc
        images = torch.from_numpy(batch.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)
        features = self.backbone(images)
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=50, pad_value=0.0), {}


def yolox_s(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 0,
        num_classes: int = 8,
        base_depth: int = 1,
        base_channel: int = 32,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (128, 256, 512),
        mid_channel: int = 128,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> YOLOXSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return YOLOXSystem(**__d)


def yolox_m(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 0,
        num_classes: int = 8,
        base_depth: int = 2,
        base_channel: int = 48,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (192, 384, 768),
        mid_channel: int = 192,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.4,
        ignore_value: float = 1.7,
        **kwargs
) -> YOLOXSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return YOLOXSystem(**__d)


def yolox_l(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 0,
        num_classes: int = 8,
        base_depth: int = 3,
        base_channel: int = 64,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (256, 512, 1024),
        mid_channel: int = 256,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.5,
        ignore_value: float = 1.6,
        **kwargs
) -> YOLOXSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return YOLOXSystem(**__d)
