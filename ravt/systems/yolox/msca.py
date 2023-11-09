import itertools
import random
from typing import Optional, Tuple, Literal, List, Union, Dict

import kornia.augmentation as ka
import numpy as np
import torch
from torch import nn

from ravt.core.base_classes import BaseDataSource, BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType, BatchDict, PredDict, LossDict
from ravt.core.utils.array_operations import clip_or_pad_along

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer, concat_pyramids
from .blocks.backbones import YOLOXPAFPNBackbone, DAMOBackbone
from .blocks.necks import TA5Neck
from .blocks.heads import TALHead
from .blocks.schedulers import StreamYOLOScheduler
from ..data_samplers import YOLOXDataSampler
from ..metrics import COCOEvalMAPMetric
from ..transforms import KorniaAugmentation


class MSCASystem(YOLOXBaseSystem):
    def __init__(
            self,
            data_source: Optional[BaseDataSource] = None,
            strategy: Optional[BaseSAPStrategy] = None,

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
            backbone: Literal['pafpn', 'drfpn'] = 'drfpn',

            # neck type
            neck_act_type: Literal['none', 'relu', 'elu', '1lu'] = 'none',

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
        self.save_hyperparameters(ignore=['kwargs', 'data_source', 'strategy'])

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams) if backbone == 'pafpn' else DAMOBackbone(**self.hparams),
            neck=TA5Neck(**self.hparams),
            head=TALHead(**self.hparams),
            with_bbox_0_train=True,
            data_source=data_source,
            data_sampler=YOLOXDataSampler(
                1, [*past_time_constant, 0], future_time_constant,
                [[*past_time_constant, 0]], [[0, *future_time_constant]]
            ),
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
            metric=COCOEvalMAPMetric(future_time_constant=future_time_constant),
            strategy=strategy,
        )

    def inference_impl(
            self,
            image: ImageInferenceType,
            buffer: Optional[YOLOXBuffer],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BBoxesInferenceType, Optional[Dict]]:
        images = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)
        features = self.backbone(images)

        if buffer is not None:
            past_indices = buffer['prev_indices']
            past_features = buffer['prev_features']
        else:
            past_indices = []
            past_features = []

        if past_time_constant is None:
            default_max_past = 3
            default_interval = 1
            ptc = [p-default_interval for p in past_indices[-default_max_past:]]
            pf = past_features[-default_max_past:]
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
            features = concat_pyramids([features]*len(future_time_constant))
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=50, pad_value=0.0), new_buffer
    
    def forward_impl(
            self,
            batch: BatchDict,
    ) -> Union[PredDict, LossDict]:
        if self.training:
            # random mask past and future
            TP = batch['image']['clip_id'].size(1) - 1
            image_masks = random.sample(range(TP), k=random.randint(1, TP)) + [TP]
            image_masks.sort()
            for k in batch['image'].keys():
                batch['image'][k] = batch['image'][k][:, image_masks]

            TF = batch['bbox']['clip_id'].size(1) - 1
            bbox_masks = [0] + random.sample(range(1, TF+1), k=random.randint(1, TF))
            bbox_masks.sort()
            for k in batch['bbox'].keys():
                batch['bbox'][k] = batch['bbox'][k][:, bbox_masks]
        return super().forward_impl(batch)

    def configure_optimizers(self):
        p_normal, p_wd, p_normal_lr, p_wd_lr, p_add = [], [], [], [], []
        all_parameters = []
        all_parameters.extend(self.backbone.named_modules(prefix='backbone'))
        all_parameters.extend(self.neck.named_modules(prefix='neck'))
        all_parameters.extend(self.head.named_modules(prefix='head'))
        for k, v in all_parameters:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                if 'neck' in k or 'head' in k:
                    p_normal_lr.append(v.bias)
                else:
                    p_normal.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)) or 'bn' in k:
                if 'neck' in k or 'head' in k:
                    p_normal_lr.append(v.weight)
                else:
                    p_normal.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                if 'neck' in k or 'head' in k:
                    p_wd_lr.append(v.weight)
                else:
                    p_wd.append(v.weight)  # apply decay
            # neck
            if hasattr(v, 'p_attn'):
                p_add.append(v.p_attn)
        optimizer = torch.optim.SGD(
            [
                {'params': p_normal},
                {'params': p_wd, 'weight_decay': self.hparams.weight_decay},
                {'params': p_normal_lr, 'lr': self.hparams.lr * 5},
                {'params': p_wd_lr, 'lr': self.hparams.lr * 5, 'weight_decay': self.hparams.weight_decay},
                {'params': p_add, 'lr': self.hparams.lr * 30}
            ],
            lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True
        )
        scheduler = StreamYOLOScheduler(
            optimizer,
            int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]


def msca_s(
        data_source: Optional[BaseDataSource] = None,
        strategy: Optional[BaseSAPStrategy] = None,

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
        backbone: Literal['pafpn', 'drfpn'] = 'drfpn',
        neck_act_type: Literal['none', 'relu', 'elu', '1lu'] = 'none',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> MSCASystem:
    return MSCASystem(
        data_source=data_source,
        strategy=strategy,
        num_classes=num_classes,
        past_time_constant=past_time_constant,
        future_time_constant=future_time_constant,
        base_depth=base_depth,
        base_channel=base_channel,
        base_neck_depth=base_neck_depth,
        hidden_ratio=hidden_ratio,
        strides=strides,
        in_channels=in_channels,
        mid_channel=mid_channel,
        depthwise=depthwise,
        act=act,
        max_objs=max_objs,
        backbone=backbone,
        neck_act_type=neck_act_type,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )
