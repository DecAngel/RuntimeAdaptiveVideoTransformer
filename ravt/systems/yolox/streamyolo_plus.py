from typing import Optional, Tuple, Literal, List, Dict

import kornia.augmentation as ka
import numpy as np
import torch

from ravt.core.base_classes import BaseDataSource, BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType
from ravt.core.utils.array_operations import clip_or_pad_along

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer, concat_pyramids
from .blocks.backbones import YOLOXPAFPNBackbone
from .blocks.necks import DFP
from .blocks.heads import TALHead
from ..data_samplers import YOLOXDataSampler
from ..metrics import COCOEvalMAPMetric
from ..transforms import KorniaAugmentation


class StreamYOLOPlusSystem(YOLOXBaseSystem):
    def __init__(
            self,
            data_source: Optional[BaseDataSource] = None,
            strategy: Optional[BaseSAPStrategy] = None,

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
        self.save_hyperparameters(ignore=['kwargs', 'data_source', 'strategy'])
        # image_samples = [[-i, 0] for i in range(3)]
        # bbox_samples = [[0, i] for i in range(3)]

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams),
            neck=DFP(**self.hparams),
            head=TALHead(**self.hparams),
            with_bbox_0_train=True,
            data_source=data_source,
            # data_sampler=YOLOXDataSampler(1, [-predict_num, 0], [predict_num], image_samples, bbox_samples),
            data_sampler=YOLOXDataSampler(1, [-predict_num, 0], [predict_num], [[-3, -2, -1, 0]], [[0, 1, 2, 3]]),
            transform=KorniaAugmentation(
                train_aug=ka.VideoSequential(ka.RandomHorizontalFlip()),
                train_resize=ka.VideoSequential(
                    *[ka.Resize((h, w)) for h, w in [
                        (496, 800),
                        (496, 816),
                        (512, 832),
                        (528, 848),
                        (528, 864),
                        (544, 880),
                        (560, 896),
                        (560, 912),
                        (576, 928),
                        (576, 944),
                        (592, 960),
                        (608, 976),
                        (608, 992),
                        (624, 1008),
                        (640, 1024),
                        (640, 1040),
                        (656, 1056),
                        (656, 1072),
                        (672, 1088),
                        (688, 1104),
                        (688, 1120),
                    ]],
                    random_apply=1,
                ),
                eval_aug=None,
                eval_resize=ka.VideoSequential(ka.Resize((600, 960))),
            ),
            metric=COCOEvalMAPMetric(future_time_constant=[predict_num]),
            strategy=strategy,
        )

    def inference_impl(
            self,
            image: ImageInferenceType,
            buffer: Optional[YOLOXBuffer],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BBoxesInferenceType, Optional[Dict]]:
        # Ignore ptc and ftc
        images = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)
        features = self.backbone(images)

        # Buffer and neck
        new_buffer = {
            'prev_features': [features],
        }
        if buffer is None or 'prev_features' not in buffer:
            features = self.neck(concat_pyramids([features, features]))
        else:
            features = self.neck(concat_pyramids([*buffer['prev_features'], features]))
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=50, pad_value=0.0), new_buffer


def streamyolo_plus_s(
        data_source: Optional[BaseDataSource] = None,
        strategy: Optional[BaseSAPStrategy] = None,

        predict_num: int = 1,
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
) -> StreamYOLOPlusSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return StreamYOLOPlusSystem(**__d)


def streamyolo_plus_m(
        data_source: Optional[BaseDataSource] = None,
        strategy: Optional[BaseSAPStrategy] = None,

        predict_num: int = 1,
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
) -> StreamYOLOPlusSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return StreamYOLOPlusSystem(**__d)


def streamyolo_plus_l(
        data_source: Optional[BaseDataSource] = None,
        strategy: Optional[BaseSAPStrategy] = None,

        predict_num: int = 1,
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
) -> StreamYOLOPlusSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return StreamYOLOPlusSystem(**__d)