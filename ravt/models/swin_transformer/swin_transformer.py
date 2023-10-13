from typing import Optional, Union, Dict, Tuple

import torch
import torch.nn as nn
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch.optim import AdamW

from .blocks import SwinTransformerBackbone, YOLOXHead, SwinScheduler, StreamYOLOScheduler
from ..transforms import KorniaAugmentation
from ..metrics import COCOEvalMAPMetric
from ravt.core.constants import (
    BatchDict, PredDict, LossDict, BatchKeys,
)
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.base_classes import BaseSystem


class SwinTransformerSystem(BaseSystem):
    def __init__(
            self,
            # structural parameters
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 18, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: int = 7,
            pretrain_img_size: Tuple[int, int] = (600, 960),
            ape: bool = False,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.2,
            patch_norm: bool = True,
            use_checkpoint: bool = False,
            frozen_stages: int = -1,

            num_classes: int = 8,
            max_objs: int = 100,
            strides: Tuple[int, ...] = (4, 8, 16, 32),
            in_channels: Tuple[int, ...] = (96, 192, 384, 768),
            mid_channel: int = 128,

            # predict parameters
            predict_num: int = 0,

            # preprocess parameters
            norm_mean: Tuple[float, ...] = (0.4831, 0.4542, 0.4044),
            norm_std: Tuple[float, ...] = (0.2281, 0.2231, 0.2241),

            # postprocess parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,

            # learning rate parameters
            swin_lr: float = 0.0001,
            swin_weight_decay: float = 0.05,
            yolox_lr: float = 0.001,
            yolox_momentum: float = 0.9,
            yolox_weight_decay: float = 5e-4,

            **kwargs,
    ):
        super().__init__(
            preprocess=KorniaAugmentation(
                train_aug=ka.VideoSequential(ka.RandomHorizontalFlip()),
                train_resize=ka.VideoSequential(
                    # *[nn.Sequential(
                    #     ka.Resize((h, w)), ka.Normalize(mean=list(norm_mean), std=list(norm_std))
                    # ) for h, w in zip(range(540, 661, 10), range(864, 1057, 16))],
                    # random_apply=1,
                    ka.Resize((600, 960)),
                    ka.Normalize(mean=list(norm_mean), std=list(norm_std)),
                ),
                eval_aug=None,
                eval_resize=ka.VideoSequential(
                    ka.Resize((600, 960)),
                    ka.Normalize(mean=list(norm_mean), std=list(norm_std)),
                ),
            ),
            metric=COCOEvalMAPMetric(),
        )
        self.save_hyperparameters(ignore=['kwargs'])

        self.backbone = SwinTransformerBackbone(**self.hparams)
        self.head = YOLOXHead(**self.hparams)
        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)
        self.register_buffer('cp', tensor=torch.tensor(self.hparams.predict_num, dtype=torch.int32), persistent=False)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.head.apply(init_yolo)

    @property
    def example_input_array(self) -> Tuple[BatchDict]:
        return {
            'image': {
                'image_id': torch.arange(0, 4, dtype=torch.int32).reshape(4, 1),
                'seq_id': torch.ones(4, 1, dtype=torch.int32),
                'frame_id': torch.arange(0, 4, dtype=torch.int32).reshape(4, 1),
                'image': torch.zeros(4, 1, 3, 600, 960, dtype=torch.uint8),
                'original_size': torch.ones(4, 1, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                'image_id': torch.arange(0, 4, dtype=torch.int32).reshape(4, 1),
                'seq_id': torch.ones(4, 1, dtype=torch.int32),
                'frame_id': torch.arange(0, 4, dtype=torch.int32).reshape(4, 1),
                'coordinate': torch.zeros(4, 1, 100, 4, dtype=torch.float32),
                'label': torch.zeros(4, 1, 100, dtype=torch.int32),
                'probability': torch.zeros(4, 1, 100, dtype=torch.float32),
            }
        },

    @property
    def required_keys_train(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.hparams.predict_num,
            'image': [0],
            'bbox': [self.hparams.predict_num],
        }

    @property
    def required_keys_eval(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.hparams.predict_num,
            'image': [0],
            # 'bbox': [self.hparams.predict_num],
        }

    @property
    def produced_keys(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.hparams.predict_num,
            'bbox': [self.hparams.predict_num],
        }

    def pth_adapter(self, state_dict: Dict) -> Dict:
        s = self.state_dict()
        new_ckpt = {}
        shape_mismatches = []
        for k, v in state_dict['state_dict'].items():
            if k in s and s[k].shape != v.shape:
                shape_mismatches.append(k)
            else:
                new_ckpt[k] = v
        if len(shape_mismatches) > 0:
            logger.warning(f'Ignoring params with mismatched shape: {shape_mismatches}')
        return new_ckpt

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        images: Float[torch.Tensor, 'B Ti C H W'] = batch['image']['image'] / self.c255
        coordinates: Optional[Float[torch.Tensor, 'B Tb O C']] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B Tb O']] = batch['bbox']['label'] if 'bbox' in batch else None

        # BGR2RGB
        images = torch.flip(images, dims=[2])
        image_0, = images.unbind(1)
        feature_p = self.backbone(image_0)

        if self.training:
            coordinate_p, = coordinates.unbind(1)
            label_p, = labels.unbind(1)
            loss_dict = self.head(
                feature_p,
                gt_coordinates=coordinate_p,
                gt_labels=label_p,
                shape=image_0.shape[-2:]
            )
            return loss_dict
        else:
            pred_dict = self.head(feature_p, shape=image_0.shape[-2:])
            return {
                'bbox': {
                    'image_id': batch['image']['image_id'] + self.cp,
                    'seq_id': batch['image']['seq_id'],
                    'frame_id': batch['image']['frame_id'] + self.cp,
                    'coordinate': pred_dict['pred_coordinates'].unsqueeze(1),
                    'label': pred_dict['pred_labels'].unsqueeze(1),
                    'probability': pred_dict['pred_probabilities'].unsqueeze(1),
                }
            }

    def configure_optimizers(self):
        epoch_steps = int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)

        # swin optimization
        p_swin_1, p_swin_2 = [], []
        for k, v in self.named_modules():
            if (
                'absolute_pos_embed' in k or
                'relative_position_bias_table' in k or
                'norm' in k
            ):
                p_swin_2.extend(v.parameters(recurse=False))
            else:
                p_swin_1.extend(v.parameters(recurse=False))
        optimizer_swin = AdamW(
            p_swin_1, lr=self.hparams.swin_lr, betas=(0.9, 0.999), weight_decay=self.hparams.swin_weight_decay
        )
        optimizer_swin.add_param_group({"params": p_swin_2, "weight_decay": 0.0})
        scheduler_swin = SwinScheduler(optimizer_swin, epoch_steps)

        # yolox optimization
        """
        p_yolox_bias, p_yolox_norm, p_yolox_weight = [], [], []
        for k, v in self.head.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                p_yolox_bias.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                p_yolox_norm.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                p_yolox_weight.append(v.weight)  # apply decay
        optimizer_yolox = torch.optim.SGD(p_yolox_norm, lr=self.hparams.yolox_lr, momentum=self.hparams.yolox_momentum, nesterov=True)
        optimizer_yolox.add_param_group({"params": p_yolox_weight, "weight_decay": self.hparams.yolox_weight_decay})
        optimizer_yolox.add_param_group({"params": p_yolox_bias})
        scheduler_yolox = StreamYOLOScheduler(optimizer_yolox, epoch_steps)
        
        return (
            [optimizer_swin, optimizer_yolox],
            [
                {'scheduler': scheduler_swin, 'interval': 'step', 'name': 'Swin_lr'},
                {'scheduler': scheduler_yolox, 'interval': 'step', 'name': 'YOLOX_lr'},
            ]
        )
        """
        return (
            [optimizer_swin],
            [
                {'scheduler': scheduler_swin, 'interval': 'step', 'name': 'Swin_lr'},
            ]
        )


def swin_transformer_small_patch4_window7(
        num_classes: int = 8,
        predict_num: int = 0,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 18, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        pretrain_img_size: Tuple[int, int] = (600, 960),
        ape: bool = False,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        frozen_stages: int = -1,
        max_objs: int = 100,
        strides: Tuple[int, ...] = (4, 8, 16, 32),
        in_channels: Tuple[int, ...] = (96, 192, 384, 768),
        mid_channel: int = 128,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        swin_lr: float = 0.0001,
        swin_weight_decay: float = 0.05,
        yolox_lr: float = 0.001,
        yolox_momentum: float = 0.9,
        yolox_weight_decay: float = 5e-4,
        **kwargs,
) -> SwinTransformerSystem:
    return SwinTransformerSystem(
        num_classes=num_classes,
        predict_num=predict_num,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        pretrain_img_size=pretrain_img_size,
        ape=ape,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        frozen_stages=frozen_stages,
        max_objs=max_objs,
        strides=strides,
        in_channels=in_channels,
        mid_channel=mid_channel,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        swin_lr=swin_lr,
        swin_weight_decay=swin_weight_decay,
        yolox_lr=yolox_lr,
        yolox_momentum=yolox_momentum,
        yolox_weight_decay=yolox_weight_decay,
        **kwargs,
    )
