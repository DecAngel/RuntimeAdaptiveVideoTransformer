from typing import Optional, Dict, Tuple, Literal, Union, List

import cv2
import numpy as np
import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch import nn

from .blocks import (
    YOLOXPAFPNBackbone, TALHead, MSCAScheduler, StreamYOLOScheduler, TANeck, YOLOXHead, TA5Neck
)
from ..transforms import KorniaAugmentation
from ..metrics import COCOEvalMAPMetric
from ravt.core.constants import (
    BatchDict, PredDict, LossDict, BatchKeys,
)
from ravt.core.base_classes import BaseSystem
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ...core.utils.array_operations import clip_or_pad_along


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
            neck_type: Literal['ta', 'ta5'] = 'ta',

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
        past_time_constant = past_time_constant or [-1]
        future_time_constant = future_time_constant or [1]
        pos_0 = -min(past_time_constant)
        pos_ptc = [c+pos_0 for c in past_time_constant]
        pos_ftc = [c+pos_0 for c in future_time_constant]
        self.save_hyperparameters(ignore=['kwargs'])
        self.hparams.pos_0 = pos_0
        self.hparams.pos_ptc = pos_ptc
        self.hparams.pos_ftc = pos_ftc

        self.backbone = YOLOXPAFPNBackbone(**self.hparams)
        if neck_type == 'ta':
            self.neck = TANeck(
                in_channels,
                self.hparams.past_time_constant,
                self.hparams.future_time_constant,
            )
        elif neck_type == 'ta5':
            self.neck = TA5Neck(
                in_channels,
                self.hparams.past_time_constant,
                self.hparams.future_time_constant,
            )
        else:
            raise ValueError(f'Unsupported neck_type: {neck_type}')
        self.head = TALHead(**self.hparams)
        self.register_buffer(
            'c_pos_ftc',
            torch.tensor([pos_ftc], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            'c_pos_ptc',
            torch.tensor([pos_ptc], dtype=torch.int32),
            persistent=False,
        )

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_yolo)

    @property
    def example_input_array(self) -> Tuple[BatchDict]:
        b = 2
        p = len(self.hparams.past_time_constant) + 1
        f = len(self.hparams.future_time_constant) + 1
        return {
            'image': {
                'image_id': torch.arange(0, b*p, dtype=torch.int32).reshape(b, p),
                'seq_id': torch.ones(b, p, dtype=torch.int32),
                'frame_id': torch.arange(0, b*p, dtype=torch.int32).reshape(b, p),
                'clip_id': torch.arange(0, p, dtype=torch.int32).unsqueeze(0).expand(b, -1),
                'image': torch.randint(0, 255, (b, p, 3, 600, 960), dtype=torch.uint8),
                'original_size': torch.ones(b, p, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                'image_id': torch.arange(0, b*f, dtype=torch.int32).reshape(b, f),
                'seq_id': torch.ones(b, f, dtype=torch.int32),
                'frame_id': torch.arange(0, b*f, dtype=torch.int32).reshape(b, f),
                'clip_id': torch.arange(p, p+f-1, dtype=torch.int32).unsqueeze(0).expand(b, -1),
                'coordinate': torch.zeros(b, f, 100, 4, dtype=torch.float32),
                'label': torch.zeros(b, f, 100, dtype=torch.int32),
                'probability': torch.zeros(b, f, 100, dtype=torch.float32),
            }
        },

    @property
    def required_keys_train(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': max(self.hparams.pos_ftc)+1,
            'image': self.hparams.pos_ptc+[self.hparams.pos_0],
            'bbox': [self.hparams.pos_0]+self.hparams.pos_ftc,
        }

    @property
    def required_keys_eval(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': max(self.hparams.pos_ftc)+1,
            'image': self.hparams.pos_ptc+[self.hparams.pos_0],
            # 'bbox': [self.hparams.pos_0]+self.hparams.pos_ftc,
        }

    @property
    def produced_keys(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': max(self.hparams.pos_ftc)+1,
            # 'image': self.hparams.pos_ptc+[self.hparams.pos_0],
            'bbox': self.hparams.pos_ftc,
        }

    def pth_adapter(self, state_dict: Dict) -> Dict:
        new_ckpt = {}
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
            new_ckpt[k] = v
        return new_ckpt

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        images: Float[torch.Tensor, 'B TP0 C H W'] = batch['image']['image'].float()
        coordinates: Optional[Float[torch.Tensor, 'B TF0 O C']] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B TF0 O']] = batch['bbox']['label'] if 'bbox' in batch else None

        features_p = self.backbone(images)
        features_f = self.neck(features_p)
        if self.training:
            loss_dict = self.head(
                features_f,
                gt_coordinates=coordinates,
                gt_labels=labels,
                shape=images.shape[-2:]
            )
            return loss_dict
        else:
            pred_dict = self.head(features_f, shape=images.shape[-2:])
            return {
                'bbox': {
                    'image_id': batch['image']['image_id'][:, :1] + self.c_pos_ftc,
                    'seq_id': batch['image']['seq_id'][:, :1],
                    'frame_id': batch['image']['frame_id'][:, :1] + self.c_pos_ftc,
                    'clip_id': batch['image']['clip_id'][:, :1] + self.c_pos_ftc,
                    'coordinate': pred_dict['pred_coordinates'],
                    'label': pred_dict['pred_labels'],
                    'probability': pred_dict['pred_probabilities'],
                }
            }

    def inference_impl(
            self, image: np.ndarray, buffer: Optional[Dict] = None,
            past_time_constant: Optional[List[int]] = None, future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        past_time_constant = past_time_constant or [-1]
        future_time_constant = future_time_constant or [1]

        image = cv2.resize(image, (960, 600))
        images = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)

        features = self.backbone(images)

        if buffer is None or 'prev_features' not in buffer:
            new_buffer = {
                'prev_indices': [-1],
                'prev_features': [features],
            }
        else:
            past_time_constant = [i for i in past_time_constant if i in buffer['prev_indices']]
            past_time_indices = [idx for idx, i in enumerate(buffer['prev_indices']) if i in past_time_constant]
            prev_features = [buffer['prev_features'][idx] for idx in past_time_indices]

            new_buffer = {
                'prev_indices': [i-1 for i in past_time_constant] + [-1],
                'prev_features': prev_features + [features],
            }
            features = self.neck(
                tuple(
                    torch.cat([f[i] for f in new_buffer['prev_features']], dim=1)
                    for i in range(len(features))
                ), past_time_constant, future_time_constant
            )
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0, 0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, 0, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, 0, :, None].astype(float),
        ], axis=1), axis=0, fixed_length=50, pad_value=0.0), new_buffer

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
            if self.hparams.neck_type == 'ta' and hasattr(v, 'attn_weight_ta'):
                p_add.append(v.attn_weight_ta)
            elif self.hparams.neck_type == 'ta5' and hasattr(v, 'p_attn'):
                p_add.append(v.p_attn)
        optimizer = torch.optim.SGD(
            [
                {'params': p_normal},
                {'params': p_wd, 'weight_decay': self.hparams.weight_decay},
                {'params': p_normal_lr, 'lr': self.hparams.lr*5},
                {'params': p_wd_lr, 'lr': self.hparams.lr*5, 'weight_decay': self.hparams.weight_decay},
                {'params': p_add, 'lr': self.hparams.lr*30}
            ],
            lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True
        )
        scheduler = StreamYOLOScheduler(optimizer, int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]


def msca_s(
        past_time_constant: List[int] = None,
        future_time_constant: List[int] = None,
        num_classes: int = 8,
        base_depth: int = 1,
        base_channel: int = 32,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (128, 256, 512),
        mid_channel: int = 128,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        neck_type: Literal['ta', 'ta5'] = 'ta',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> MSCASystem:
    return MSCASystem(
        past_time_constant=past_time_constant,
        future_time_constant=future_time_constant,
        num_classes=num_classes,
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
