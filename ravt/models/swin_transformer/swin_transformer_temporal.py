from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Callable

import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from .blocks import SwinTransformerBackbone, YOLOXHead
from ..augmentation import KorniaAugmentation
from ..metrics import COCOEvalMAPMetric
from ravt.protocols.classes import BaseModel
from ravt.protocols.structures import (
    BatchDict, PredDict, LossDict, BatchKeys,
)
from ravt.core.utils.lightning_logger import ravt_logger as logger


class SwinTransformerTemporalSystem(BaseModel):
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

            # pretrain parameters
            pretrain_adapter: Optional[Callable[[Path], Dict]] = None,

            # preprocess parameters
            norm_mean: Tuple[float, ...] = (0.4831, 0.4542, 0.4044),
            norm_std: Tuple[float, ...] = (0.2281, 0.2231, 0.2241),

            # predict parameters
            predict_num: int = 1,

            # other parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,
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
        self.head = YOLOXHead(width_ratio=1.0, **self.hparams)
        self.register_buffer('t2', tensor=torch.tensor(2, dtype=torch.int32), persistent=False)

    @property
    def example_input_array(self) -> Tuple[BatchDict]:
        b = 4
        p = self.hparams.predict_num
        meta = {
            'image_id': torch.arange(0, b*(p+1), dtype=torch.int32).reshape(b, p+1),
            'seq_id': torch.ones(b, p+1, dtype=torch.int32),
            'frame_id': torch.arange(0, b*(p+1), dtype=torch.int32).reshape(b, p+1),
        }
        return {
            'image': {
                **meta,
                'image': torch.zeros(b, p+1, 3, 600, 960, dtype=torch.uint8),
                'original_size': torch.ones(b, p+1, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                **meta,
                'coordinate': torch.zeros(b, p+1, 100, 4, dtype=torch.float32),
                'label': torch.zeros(b, p+1, 100, dtype=torch.int32),
                'probability': torch.zeros(b, p+1, 100, dtype=torch.float32),
            }
        },

    @property
    def required_keys_train(self) -> BatchKeys:
        p = self.hparams.predict_num
        return {
            'interval': 1,
            'margin': 2*p,
            'image': list(range(0, p+1)),
            'bbox': list(range(p, 2*p+1)),
        }

    @property
    def required_keys_eval(self) -> BatchKeys:
        p = self.hparams.predict_num
        return {
            'interval': 1,
            'margin': 2*p,
            'image': list(range(0, p+1)),
            'bbox': list(range(p, 2*p+1)),
        }

    @property
    def produced_keys(self) -> BatchKeys:
        p = self.hparams.predict_num
        return {
            'interval': 1,
            'margin': 2*p,
            'bbox': [2*p],
        }

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        images: Float[torch.Tensor, 'B Ti C H W'] = batch['image']['image']
        coordinates: Optional[Float[torch.Tensor, 'B Tb O C']] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B Tb O']] = batch['bbox']['label'] if 'bbox' in batch else None

        # BGR2RGB
        images = torch.flip(images, dims=[2])
        feature2 = self.backbone(images)

        if self.training:
            coordinate2 = coordinates[:, -1]
            label2 = labels[:, -1]
            loss_dict = self.head(
                feature2,
                gt_coordinates=coordinate2,
                gt_labels=label2,
                shape=images.shape[-2:]
            )
            return loss_dict
        else:
            pred_dict = self.head(feature2, shape=images.shape[-2:])
            return {
                'bbox': {
                    'image_id': batch['image']['image_id'][:, -1:] + self.hparams.predict_num,
                    'seq_id': batch['image']['seq_id'][:, -1:],
                    'frame_id': batch['image']['frame_id'][:, -1:] + self.hparams.predict_num,
                    'coordinate': pred_dict['pred_coordinates'].unsqueeze(1),
                    'label': pred_dict['pred_labels'].unsqueeze(1),
                    'probability': pred_dict['pred_probabilities'].unsqueeze(1),
                }
            }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.05)
        scheduler = StepLR(optimizer, 1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


def swin_transformer_temporal_small_patch4_window7(
        pretrained: bool = True,
        predict_num: int = 1,
        num_classes: int = 8,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        gamma: float = 0.99,
        **kwargs,
) -> SwinTransformerTemporalSystem:
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

    model = SwinTransformerTemporalSystem(
        pretrain_adapter=pth_adapter if pretrained else None,
        predict_num=predict_num,
        num_classes=num_classes,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        gamma=gamma,
        **kwargs,
    )

    return model
