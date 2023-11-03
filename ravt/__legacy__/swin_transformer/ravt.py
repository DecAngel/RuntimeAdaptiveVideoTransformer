from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Callable

import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from .blocks import RAVTBackbone, YOLOXHead
from ravt.models.transforms import KorniaAugmentation
from ravt.models.metrics import COCOEvalMAPMetric
from ravt.core.constants import (
    BatchDict, PredDict, LossDict, BatchKeys,
)
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.base_classes import BaseSystem


class RAVTSystem(BaseSystem):
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
            predict_nums: Tuple[int, ...] = (1, ),

            # preprocess parameters
            norm_mean: Tuple[float, ...] = (0.4831, 0.4542, 0.4044),
            norm_std: Tuple[float, ...] = (0.2281, 0.2231, 0.2241),

            # postprocess parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,

            # learning rate parameters
            lr: float = 0.001,
            gamma: float = 0.99,
            weight_decay: float = 0.1,
            **kwargs,
    ):
        super().__init__(
            preprocess=KorniaAugmentation(
                train_aug=ka.VideoSequential(ka.RandomHorizontalFlip()),
                train_resize=ka.VideoSequential(
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

        self.hparams.predict_nums = sorted(list(set(self.hparams.predict_nums)))
        predict_nums_with_0 = self.hparams.predict_nums
        if predict_nums_with_0[0] != 0:
            predict_nums_with_0.insert(0, 0)

        self.max_pred = max(self.hparams.predict_nums)
        self.image_indices = [self.max_pred - p for p in predict_nums_with_0]
        self.bbox_train_indices = [self.max_pred + p for p in predict_nums_with_0]
        self.bbox_eval_indices = [self.max_pred + p for p in predict_nums]
        self.backbone = RAVTBackbone(
            **self.hparams,
            neck_type='t1',
            neck_temporal_frame=tuple(self.image_indices),
        )
        self.head = YOLOXHead(**self.hparams)
        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)
        self.register_buffer(
            'cp',
            tensor=torch.tensor([2*p for p in self.hparams.predict_nums[::-1]], dtype=torch.int32),
            persistent=False
        )

    @property
    def example_input_array(self) -> Tuple[BatchDict]:
        batch_size = 2
        max_len = len(self.hparams.predict_nums) + (not self.has_0)
        return {
            'image': {
                'image_id': torch.arange(0, batch_size*max_len, dtype=torch.int32).reshape(batch_size, max_len),
                'seq_id': torch.ones(batch_size, max_len, dtype=torch.int32),
                'frame_id': torch.arange(0, batch_size*max_len, dtype=torch.int32).reshape(batch_size, max_len),
                'image': torch.zeros(batch_size, max_len, 3, 600, 960, dtype=torch.uint8),
                'original_size': torch.ones(batch_size, max_len, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                'image_id': torch.arange(0, batch_size*max_len, dtype=torch.int32).reshape(batch_size, max_len),
                'seq_id': torch.ones(batch_size, max_len, dtype=torch.int32),
                'frame_id': torch.arange(0, batch_size*max_len, dtype=torch.int32).reshape(batch_size, max_len),
                'coordinate': torch.zeros(batch_size, max_len, 100, 4, dtype=torch.float32),
                'label': torch.zeros(batch_size, max_len, 100, dtype=torch.int32),
                'probability': torch.zeros(batch_size, max_len, 100, dtype=torch.float32),
            }
        },

    @property
    def required_keys_train(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.max_pred*2,
            'image': self.image_indices + ([] if self.has_0 else [self.max_pred]),
            'bbox': self.bbox_eval_indices + ([] if self.has_0 else [self.max_pred]),
        }

    @property
    def required_keys_eval(self) -> BatchKeys:
        return {
            'interval': self.max_pred+1,
            'margin': self.max_pred*2,
            'image': self.image_indices + ([] if self.has_0 else [self.max_pred]),
            # 'bbox': self.bbox_indices + ([] if self.has_0 else [self.max_pred]),
        }

    @property
    def produced_keys(self) -> BatchKeys:
        return {
            'interval': self.max_pred+1,
            'margin': int(max(self.bbox_eval_indices)),
            'bbox': self.bbox_eval_indices,
        }

    def pth_adapter(self, state_dict: Dict) -> Dict:
        return state_dict['state_dict']

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        images: Float[torch.Tensor, 'B Ti C H W'] = batch['image']['image'] / self.c255
        coordinates: Optional[Float[torch.Tensor, 'B Tb O C']] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B Tb O']] = batch['bbox']['label'] if 'bbox' in batch else None

        # BGR2RGB
        images = torch.flip(images, dims=[2])
        features = self.backbone(images)
        features = [tuple([_f[:, i] for _f in features]) for i in range(features[0].size(1))]

        if self.training:
            losses = []
            for f, c, l in zip(
                features,
                coordinates.unbind(1),
                labels.unbind(1),
            ):
                losses.append(self.head(
                    f,
                    gt_coordinates=c,
                    gt_labels=l,
                    shape=images.shape[-2:]
                )['loss'])
            return {'loss': sum(losses)}
        else:
            pred_coordinates = []
            pred_labels = []
            pred_probabilities = []
            for f in features:
                pred_dict = self.head(f, shape=images.shape[-2:])
                pred_coordinates.append(pred_dict['pred_coordinates'])
                pred_labels.append(pred_dict['pred_labels'])
                pred_probabilities.append(pred_dict['pred_probabilities'])

            return {
                'bbox': {
                    'image_id': batch['image']['image_id'] + self.cp,
                    'seq_id': batch['image']['seq_id'],
                    'frame_id': batch['image']['frame_id'] + self.cp,
                    'coordinate': torch.stack(pred_coordinates, dim=1),
                    'label': torch.stack(pred_labels, dim=1),
                    'probability': torch.stack(pred_probabilities, dim=1),
                }
            }

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, 1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


def ravt_small_patch4_window7(
        num_classes: int = 8,
        predict_nums: Tuple[int, ...] = (1, ),
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
        lr: float = 0.001,
        gamma: float = 0.99,
        weight_decay: float = 0.1,
        **kwargs,
) -> RAVTSystem:
    return RAVTSystem(
        num_classes=num_classes,
        predict_nums=predict_nums,
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
        lr=lr,
        gamma=gamma,
        weight_decay=weight_decay,
        **kwargs,
    )
