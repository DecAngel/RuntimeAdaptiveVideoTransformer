from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Callable

import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from .blocks import SwinTransformerBackbone, YOLOXHead
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
            drop_path_rate: float = 0.2,
            patch_norm: bool = True,
            use_checkpoint: bool = False,
            frozen_stages: int = -1,

            num_classes: int = 8,
            max_objs: int = 100,
            strides: Tuple[int, ...] = (4, 8, 16, 32),
            in_channels: Tuple[int, ...] = (96, 192, 384, 768),
            mid_channel: int = 128,

            # pretrain parameters
            pretrain_adapter: Optional[Callable[[Path], Dict]] = None,

            # predict parameters
            predict_num: int = 0,

            # preprocess parameters
            norm_mean: Tuple[float, ...] = (0.4831, 0.4542, 0.4044),
            norm_std: Tuple[float, ...] = (0.2281, 0.2231, 0.2241),

            # postprocess parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,

            # learning rate parameters
            lr: float = 0.001,
            gamma: float = 0.99,
            **kwargs,
    ):
        super().__init__(
            pretrain_adapter=pretrain_adapter,
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
        self.save_hyperparameters(ignore=['kwargs', 'pretrain_adapter'])

        self.backbone = SwinTransformerBackbone(**self.hparams)
        self.head = YOLOXHead(**self.hparams)
        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)
        self.register_buffer('cp', tensor=torch.tensor(self.hparams.predict_num, dtype=torch.int32), persistent=False)

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
            'bbox': [self.hparams.predict_num],
        }

    @property
    def produced_keys(self) -> BatchKeys:
        return {
            'interval': 1,
            'margin': self.hparams.predict_num,
            'bbox': [self.hparams.predict_num],
        }

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
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.05)
        scheduler = StepLR(optimizer, 1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


def swin_transformer_small_patch4_window7(
        pretrained: bool = True,
        num_classes: int = 8,
        predict_num: int = 0,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 18, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        pretrain_img_size: Tuple[int, int] = (600, 960),
        ape: bool = False,
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
        **kwargs,
) -> SwinTransformerSystem:
    def pth_adapter(weight_pretrained_dir: Path) -> Dict:
        file = weight_pretrained_dir.joinpath('cascade_mask_rcnn_swin_small_patch4_window7.pth')
        if file.exists():
            logger.info(f'Load pretrained file cascade_mask_rcnn_swin_small_patch4_window7.pth')
            return torch.load(str(file))['state_dict']
        else:
            raise FileNotFoundError(
                f'pretrained file cascade_mask_rcnn_swin_small_patch4_window7.pth '
                f'not found in {str(weight_pretrained_dir)}!'
            )

    model = SwinTransformerSystem(
        pretrain_adapter=pth_adapter if pretrained else None,
        num_classes=num_classes,
        predict_num=predict_num,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        pretrain_img_size=pretrain_img_size,
        ape=ape,
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
        **kwargs,
    )

    return model
