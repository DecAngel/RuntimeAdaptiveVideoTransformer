from typing import Optional, Tuple, Literal

import kornia.augmentation as ka

from .yolox_base import YOLOXBaseSystem
from .blocks.backbones import YOLOXPAFPNBackbone
from .blocks.necks import DFP
from .blocks.heads import TALHead
from ..data_samplers import YOLOXDataSampler
from ..metrics import COCOEvalMAPMetric
from ..transforms import KorniaAugmentation

from ravt.core.base_classes import BaseDataSource, BaseSAPStrategy


class StreamYOLOSystem(YOLOXBaseSystem):
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

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams),
            neck=DFP(**self.hparams),
            head=TALHead(**self.hparams),
            with_bbox_0_train=True,
            data_source=data_source,
            data_sampler=YOLOXDataSampler(1, [-predict_num, 0], [predict_num], [[-predict_num, 0]], [[0, predict_num]]),
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
            metric=COCOEvalMAPMetric(),
            strategy=strategy,
        )


def streamyolo_s(
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
) -> StreamYOLOSystem:
    return StreamYOLOSystem(
        data_source=data_source,
        strategy=strategy,
        num_classes=num_classes,
        predict_num=predict_num,
        base_depth=base_depth,
        base_channel=base_channel,
        strides=strides,
        in_channels=in_channels,
        mid_channel=mid_channel,
        depthwise=depthwise,
        act=act,
        max_objs=max_objs,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )
