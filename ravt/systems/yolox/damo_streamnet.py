from typing import Optional, Tuple, Literal, List, Dict

import kornia.augmentation as ka
import numpy as np
import torch

from ravt.core.base_classes import BaseDataSource, BaseSAPStrategy, BaseDataSampler, BaseTransform, BaseMetric
from ravt.core.constants import SubsetLiteral, BatchTDict
from ravt.core.utils.array_operations import clip_or_pad_along

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer, concat_pyramids
from .blocks.backbones import DAMOBackbone
from .blocks.necks import LongShortNeck
from .blocks.heads import TALHead
from ravt.data_samplers import YOLOXDataSampler
from ravt.metrics import COCOEvalMAPMetric
from ravt.transforms import KorniaAugmentation


class DAMOStreamNetSystem(YOLOXBaseSystem):
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

            # predict parameters
            predict_num: int = 1,

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
            backbone=DAMOBackbone(**self.hparams),
            neck=LongShortNeck(**self.hparams),
            head=TALHead(**self.hparams),
            batch_size=batch_size,
            num_workers=num_workers,
            with_bbox_0_train=True,
            data_sources=data_sources,
            data_sampler=YOLOXDataSampler(1, [-3, -2, -1, 0], [predict_num], [[-3, -2, -1, 0]], [[0, predict_num]]),
            metric=COCOEvalMAPMetric(future_time_constant=[predict_num]),
            strategy=strategy,
        )

    def pth_adapter(self, state_dict: Dict) -> Dict:
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
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BatchTDict, Optional[Dict]]:
        # TODO: change
        # Ignore ptc and ftc
        images = torch.from_numpy(batch.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)
        features = self.backbone(images)

        # Buffer and neck
        if buffer is None or 'prev_features' not in buffer:
            new_buffer = {
                'prev_features': [features]*3,
            }
            features = self.neck(concat_pyramids([features]*4))
        else:
            new_buffer = {
                'prev_features': buffer['prev_features'][1:] + [features]
            }
            features = self.neck(concat_pyramids([*buffer['prev_features'], features]))
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=50, pad_value=0.0), new_buffer


def damo_streamnet_s(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 1,
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
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> DAMOStreamNetSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return DAMOStreamNetSystem(**__d)


def damo_streamnet_m(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 1,
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
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.4,
        ignore_value: float = 1.7,
        **kwargs
) -> DAMOStreamNetSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return DAMOStreamNetSystem(**__d)


def damo_streamnet_l(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 1,
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
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.5,
        ignore_value: float = 1.6,
        **kwargs
) -> DAMOStreamNetSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return DAMOStreamNetSystem(**__d)
