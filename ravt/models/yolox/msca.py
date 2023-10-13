from typing import Optional, Dict, Tuple, Literal, Union

import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch import nn

from .blocks import YOLOXPAFPNBackbone, MSCANeck, TALHead, StreamYOLOScheduler, SimpleNeck, MSCAScheduler, Simple2Neck
from ..transforms import KorniaAugmentation
from ..metrics import COCOEvalMAPMetric
from ravt.core.constants import (
    BatchDict, PredDict, LossDict, BatchKeys,
)
from ravt.core.base_classes import BaseSystem
from ravt.core.utils.lightning_logger import ravt_logger as logger


class MSCASystem(BaseSystem):
    def __init__(
            self,
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
            neck_type: Literal['mscta', 'simple', 'simple2'] = 'mscta',

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
        super().__init__(
            preprocess=KorniaAugmentation(
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
        )
        self.save_hyperparameters(ignore=['kwargs'])

        self.backbone = YOLOXPAFPNBackbone(**self.hparams)
        if neck_type == 'mscta':
            self.neck = MSCANeck(in_channels)
        elif neck_type == 'simple':
            self.neck = SimpleNeck(in_channels, time_constant=[1])
        elif neck_type == 'simple2':
            self.neck = Simple2Neck(in_channels, [-self.hparams.predict_num], [self.hparams.predict_num])
        else:
            raise ValueError(f'Unsupported neck_type: {neck_type}')
        self.head = TALHead(**self.hparams)
        self.register_buffer('cp', tensor=torch.tensor(self.hparams.predict_num, dtype=torch.int32), persistent=False)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_yolo)

    @property
    def example_input_array(self) -> Tuple[BatchDict]:
        b = 2
        p = self.hparams.predict_num + 1
        return {
            'image': {
                'image_id': torch.arange(0, b*p, dtype=torch.int32).reshape(b, p),
                'seq_id': torch.ones(b, p, dtype=torch.int32),
                'frame_id': torch.arange(0, b*p, dtype=torch.int32).reshape(b, p),
                # 'image': torch.randint(0, 255, (b, p, 3, 600, 960), dtype=torch.uint8),
                'image': torch.randint(0, 255, (b, p, 3, 600, 960), dtype=torch.uint8),
                'original_size': torch.ones(b, p, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                'image_id': torch.arange(0, b*p, dtype=torch.int32).reshape(b, p),
                'seq_id': torch.ones(b, p, dtype=torch.int32),
                'frame_id': torch.arange(0, b*p, dtype=torch.int32).reshape(b, p),
                'coordinate': torch.zeros(b, p, 100, 4, dtype=torch.float32),
                'label': torch.zeros(b, p, 100, dtype=torch.int32),
                'probability': torch.zeros(b, p, 100, dtype=torch.float32),
            }
        },

    @property
    def required_keys_train(self) -> BatchKeys:
        p = self.hparams.predict_num
        return {
            'interval': 1,
            'margin': p*2,
            'image': list(range(0, p+1)),
            'bbox': [self.hparams.predict_num, self.hparams.predict_num*2],
        }

    @property
    def required_keys_eval(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.hparams.predict_num*2,
            'image': [0, self.hparams.predict_num],
            # 'bbox': [self.hparams.predict_num, self.hparams.predict_num*2],
        }

    @property
    def produced_keys(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.hparams.predict_num*2,
            'bbox': [self.hparams.predict_num*2],
        }

    def pth_adapter(self, state_dict: Dict) -> Dict:
        s = self.state_dict()
        new_ckpt = {}
        shape_mismatches = []
        for k, v in state_dict['model'].items():
            k = k.replace('lateral_conv0', 'down_conv_5_4')
            k = k.replace('C3_p4', 'down_csp_4_4')
            k = k.replace('reduce_conv1', 'down_conv_4_3')
            k = k.replace('C3_p3', 'down_csp_3_3')
            k = k.replace('bu_conv2', 'up_conv_3_3')
            k = k.replace('C3_n3', 'up_csp_3_4')
            k = k.replace('bu_conv1', 'up_conv_4_4')
            k = k.replace('C3_n4', 'up_csp_4_5')
            k = k.replace('backbone.jian2', 'neck.convs.0')
            k = k.replace('backbone.jian1', 'neck.convs.1')
            k = k.replace('backbone.jian0', 'neck.convs.2')
            if k in s and s[k].shape != v.shape:
                shape_mismatches.append(k)
            else:
                new_ckpt[k] = v
        if len(shape_mismatches) > 0:
            logger.warning(f'Ignoring params with mismatched shape: {shape_mismatches}')
        return new_ckpt

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        images: Float[torch.Tensor, 'B Ti C H W'] = batch['image']['image'].float()
        coordinates: Optional[Float[torch.Tensor, 'B Tb O C']] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B Tb O']] = batch['bbox']['label'] if 'bbox' in batch else None

        image_0, image_p = images.unbind(1)
        feature_0, feature_p = self.backbone(image_0), self.backbone(image_p)
        feature_2p = self.neck(tuple(torch.stack([f0, fp], dim=1) for f0, fp in zip(feature_0, feature_p)))
        if self.training:
            loss_dict = self.head(
                feature_2p,
                gt_coordinates=coordinates,
                gt_labels=labels,
                shape=image_0.shape[-2:]
            )
            return loss_dict
        else:
            pred_dict = self.head(feature_2p, shape=image_0.shape[-2:])
            return {
                'bbox': {
                    'image_id': batch['image']['image_id'][:, -1:] + self.cp,
                    'seq_id': batch['image']['seq_id'][:, -1:],
                    'frame_id': batch['image']['frame_id'][:, -1:] + self.cp,
                    'coordinate': pred_dict['pred_coordinates'].unsqueeze(1),
                    'label': pred_dict['pred_labels'].unsqueeze(1),
                    'probability': pred_dict['pred_probabilities'].unsqueeze(1),
                }
            }

    def configure_optimizers(self):
        p_bias, p_norm, p_weight = [], [], []
        for k, v in self.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                p_bias.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                p_norm.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                p_weight.append(v.weight)  # apply decay
        optimizer = torch.optim.SGD(p_norm, lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True)
        optimizer.add_param_group({"params": p_weight, "weight_decay": self.hparams.weight_decay})
        optimizer.add_param_group({"params": p_bias})
        scheduler = MSCAScheduler(optimizer, int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]


def msca_s(
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
        neck_type: Literal['mscta', 'simple', 'simple2'] = 'mscta',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> MSCASystem:
    return MSCASystem(
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
        neck_type=neck_type,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )
