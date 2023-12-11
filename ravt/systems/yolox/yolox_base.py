import contextlib
import io
from typing import Optional, Dict, Tuple, Union, List, TypedDict, Sequence

import torch
import kornia.augmentation as ka
from jaxtyping import Float, Int
from pycocotools.coco import COCO
from pytorch_lightning import Callback
from torch import nn

from ravt.data_sources import ArgoverseDataSource
from ravt.data_samplers import YOLOXDataSampler
from ravt.transforms import KorniaAugmentation
from ravt.metrics import COCOEvalMAPMetric
from ravt.core.constants import (
    BatchTDict, LossDict, SubsetLiteral
)
from ravt.core.base_classes import BaseSystem, BaseDataSampler, BaseMetric, BaseTransform, BaseDataSource, BaseSAPStrategy
from ravt.core.utils.visualization import draw_feature, draw_bbox, add_clip_id
from ravt.core.utils.image_writer import ImageWriter

from .blocks import types, StreamYOLOScheduler
from ...callbacks import EMACallback


def concat_pyramids(pyramids: List[types.PYRAMID], dim: int = 1) -> types.PYRAMID:
    return tuple(
        torch.cat([
            p[i]
            for p in pyramids
        ], dim=dim)
        for i in range(len(pyramids[0]))
    )


class YOLOXBuffer(TypedDict):
    prev_indices: List[int]
    prev_features: List[types.PYRAMID]


class YOLOXBaseSystem(BaseSystem):
    def __init__(
            self,
            backbone: types.BaseBackbone,
            neck: types.BaseNeck,
            head: types.BaseHead,
            batch_size: int = 1,
            num_workers: int = 0,
            with_bbox_0_train: bool = False,
            data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
            data_sampler: Optional[BaseDataSampler] = None,
            transform: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
            strategy: Optional[BaseSAPStrategy] = None,
    ):
        data_sources = data_sources or {
            'train': ArgoverseDataSource('train'),
            'eval': ArgoverseDataSource('eval'),
            'test': ArgoverseDataSource('test'),
        }
        data_sampler = data_sampler or YOLOXDataSampler(
            interval=1,
            eval_image_clip=[0],
            eval_bbox_clip=[0],
            train_image_clip=[[0]],
            train_bbox_clip=[[0]],
        )
        transform = transform or KorniaAugmentation(
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
        )
        metric = metric or COCOEvalMAPMetric()

        super().__init__(
            data_sources=data_sources,
            data_sampler=data_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            metric=metric,
            strategy=strategy,
        )
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.with_bbox_0_train = with_bbox_0_train

        self.eia_b = 2
        self.eia_p = len(data_sampler.eval_image_clip) - 1
        self.eia_f = len(data_sampler.eval_bbox_clip)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_yolo)

    @property
    def example_input_array(self) -> Tuple[BatchTDict]:
        b = self.eia_b
        p = self.eia_p
        f = self.eia_f
        return {
            'image_id': torch.arange(0, b, dtype=torch.int32),
            'seq_id': torch.ones(b, dtype=torch.int32),
            'frame_id': torch.arange(0, b, dtype=torch.int32),
            'image': {
                'clip_id': torch.arange(-p, 1, dtype=torch.int32).unsqueeze(0).expand(b, -1),
                'image': torch.randint(0, 255, (b, p+1, 3, 600, 960), dtype=torch.uint8),
                'original_size': torch.ones(b, p+1, 2, dtype=torch.int32) * torch.tensor([1200, 1920], dtype=torch.int32),
            },
            'bbox': {
                'clip_id': torch.arange(1, f+1, dtype=torch.int32).unsqueeze(0).expand(b, -1),
                'coordinate': torch.zeros(b, f, 100, 4, dtype=torch.float32),
                'label': torch.zeros(b, f, 100, dtype=torch.int32),
            }
        },

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

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if isinstance(self.metric, COCOEvalMAPMetric):
            with contextlib.redirect_stdout(io.StringIO()):
                self.metric.coco = COCO(str(self.data_sources['eval'].ann_file))

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        if isinstance(self.metric, COCOEvalMAPMetric):
            with contextlib.redirect_stdout(io.StringIO()):
                self.metric.coco = COCO(str(self.data_sources['test'].ann_file))

    def forward_impl(
            self,
            batch: BatchTDict,
    ) -> Union[BatchTDict, LossDict]:
        images: Float[torch.Tensor, 'B TP0 C H W'] = batch['image']['image'].float()
        coordinates: Optional[Float[torch.Tensor, 'B TF0 O C']] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B TF0 O']] = batch['bbox']['label'] if 'bbox' in batch else None
        past_frame_constant = batch['image']['clip_id'][:, :-1].float()
        future_frame_constant = batch['bbox']['clip_id'][:, (int(self.with_bbox_0_train) if self.training else 0):].float()

        features_p = self.backbone(images)
        features_f = self.neck(features_p, past_frame_constant, future_frame_constant)

        if self.image_writer is not None:


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
                'image_id': batch['image_id'],
                'seq_id': batch['seq_id'],
                'frame_id': batch['frame_id'],
                'bbox': {
                    'clip_id': batch['bbox']['clip_id'],
                    'coordinate': pred_dict['pred_coordinates'],
                    'label': pred_dict['pred_labels'],
                    'probability': pred_dict['pred_probabilities'],
                },
            }

    def configure_optimizers(self):
        p_bias, p_norm, p_weight = [], [], []
        all_parameters = []
        all_parameters.extend(self.backbone.named_modules())
        all_parameters.extend(self.head.named_modules())
        for k, v in all_parameters:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                p_bias.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                p_norm.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                p_weight.append(v.weight)  # apply decay
        optimizer = torch.optim.SGD(p_norm, lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True)
        optimizer.add_param_group({"params": p_weight, "weight_decay": self.hparams.weight_decay})
        optimizer.add_param_group({"params": p_bias})
        scheduler = StreamYOLOScheduler(
            optimizer,
            int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [
            EMACallback(),
        ]
