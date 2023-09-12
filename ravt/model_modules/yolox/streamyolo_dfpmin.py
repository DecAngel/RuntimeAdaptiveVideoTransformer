from pathlib import Path
from typing import Optional, Dict, Tuple, Literal, Callable, Union

import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch import nn

from .blocks import YOLOXPAFPNBackbone, DFPMIN, TALHead, StreamYOLOScheduler, get_pth_adapter
from ..augmentation import KorniaAugmentation
from ..metrics import COCOEvalMAPMetric
from ravt.protocols.classes import BaseModel
from ravt.protocols.structures import (
    BatchDict, PredDict, LossDict, BatchKeys,
)


class StreamYOLODFPMINSystem(BaseModel):
    def __init__(
            self,
            # structural parameters
            depth_ratio: float = 0.33,
            width_ratio: float = 0.5,
            depthwise: bool = False,
            act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',

            num_classes: int = 8,
            max_objs: int = 100,
            strides: Tuple[int, ...] = (8, 16, 32),
            in_channels: Tuple[int, ...] = (256, 512, 1024),

            # pretrain parameters
            pretrain_adapter: Optional[Callable[[Path], Dict]] = None,

            # other parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,
            lr: float = 0.001,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,

            **kwargs,
    ):
        super().__init__(
            pretrain_adapter=pretrain_adapter,
            preprocess=KorniaAugmentation(
                train_aug=ka.VideoSequential(ka.RandomHorizontalFlip()),
                train_resize=ka.VideoSequential(
                    *[ka.Resize((h, w)) for h, w in zip(range(540, 661, 10), range(864, 1057, 16))],
                    random_apply=1,
                ),
                eval_aug=None,
                eval_resize=ka.VideoSequential(ka.Resize((600, 960))),
            ),
            metric=COCOEvalMAPMetric(),
        )
        self.save_hyperparameters(ignore=['kwargs', 'pretrain_adapter'])
        self.backbone = YOLOXPAFPNBackbone(**self.hparams)
        self.neck = DFPMIN(**self.hparams)
        self.head = TALHead(**self.hparams)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_yolo)

    @property
    def example_input_array(self) -> Tuple[BatchDict]:
        return {
            'image': {
                'image_id': torch.arange(0, 8, dtype=torch.int32).reshape(4, 2),
                'seq_id': torch.ones(4, 2, dtype=torch.int32),
                'frame_id': torch.arange(0, 8, dtype=torch.int32).reshape(4, 2),
                'image': torch.rand(4, 2, 3, 600, 960, dtype=torch.float32),
                'original_size': torch.ones(4, 2, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                'image_id': torch.arange(0, 8, dtype=torch.int32).reshape(4, 2),
                'seq_id': torch.ones(4, 2, dtype=torch.int32),
                'frame_id': torch.arange(0, 8, dtype=torch.int32).reshape(4, 2),
                'coordinate': torch.zeros(4, 2, 100, 4, dtype=torch.float32),
                'label': torch.zeros(4, 2, 100, dtype=torch.int32),
                'probability': torch.zeros(4, 2, 100, dtype=torch.float32),
            }
        },

    @property
    def required_keys_train(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': 3,
            'image': [0, 1],
            'bbox': [1, 2],
        }

    @property
    def required_keys_eval(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': 3,
            'image': [0, 1],
            'bbox': [1, 2]
        }

    @property
    def produced_keys(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': 3,
            'bbox': [2],
        }

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        images: Float[torch.Tensor, 'B Ti C H W'] = batch['image']['image'].clone()
        coordinates: Optional[Float[torch.Tensor, 'B Tb O C']] = batch['bbox']['coordinate'].clone() if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B Tb O']] = batch['bbox']['label'].clone() if 'bbox' in batch else None

        images *= 255.0

        image0, image1 = images.unbind(1)
        feature0, feature1 = self.backbone(image0), self.backbone(image1)
        feature2 = self.neck((feature0, feature1))
        if self.training:
            loss_dict = self.head(
                feature2,
                gt_coordinates=coordinates,
                gt_labels=labels,
                shape=image1.shape[-2:]
            )
            return loss_dict
        else:
            pred_dict = self.head(feature2, shape=image1.shape[-2:])
            return {
                'bbox': {
                    'image_id': torch.add(batch['image']['image_id'][:, -1:], 1),
                    'seq_id': batch['image']['seq_id'][:, -1:],
                    'frame_id': torch.add(batch['image']['frame_id'][:, -1:], 1),
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
        scheduler = StreamYOLOScheduler(optimizer, int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]


def streamyolo_dfpmin_s(
        pretrained: bool = True,
        num_classes: int = 8,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
) -> StreamYOLODFPMINSystem:
    return StreamYOLODFPMINSystem(
        num_classes=num_classes,
        pretrain_adapter=get_pth_adapter('yolox_s.pth') if pretrained else None,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
