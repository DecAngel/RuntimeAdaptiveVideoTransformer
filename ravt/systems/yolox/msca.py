import random
from typing import Optional, Tuple, Literal, List, Union, Dict

import kornia.augmentation as ka
import numpy as np
import torch
from torch import nn

from ravt.core.base_classes import BaseDataSource, BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType, BatchTDict, LossDict
from ravt.core.utils.array_operations import clip_or_pad_along
from ravt.core.utils.lightning_logger import ravt_logger as logger

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer, concat_pyramids
from .blocks.backbones import YOLOXPAFPNBackbone, DAMOBackbone
from .blocks.necks import TANeck, TA2Neck, TA3Neck
from .blocks.heads import TALHead
from .blocks.schedulers import StreamYOLOScheduler, MSCAScheduler
from ravt.data_samplers import YOLOXDataSampler
from ravt.metrics import COCOEvalMAPMetric
from ravt.transforms import KorniaAugmentation


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
            backbone: Literal['pafpn', 'drfpn'] = 'pafpn',

            # neck type
            neck_type: Literal['ta', 'ta2', 'ta3'] = 'ta',
            neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'relu',
            neck_p_init: Union[float, Literal['uniform', 'normal'], None] = 0.0,
            neck_tpe_merge: Literal['add', 'mul'] = 'add',
            neck_dropout: float = 0.0,

            # train type
            train_mask: bool = False,
            train_scheduler: Literal['yolox', 'msca'] = 'yolox',

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

        if neck_type == 'ta':
            NECK = TANeck
        elif neck_type == 'ta2':
            NECK = TA2Neck
        elif neck_type == 'ta3':
            NECK = TA3Neck
        else:
            raise ValueError(f'neck type {neck_type} not supported!')

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams) if backbone == 'pafpn' else DAMOBackbone(**self.hparams),
            neck=NECK(**self.hparams),
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

    def pth_adapter(self, state_dict: Dict) -> Dict:
        if self.hparams.backbone == 'pafpn':
            return super().pth_adapter(state_dict)
        else:
            new_ckpt = {}
            for k, v in state_dict['model'].items():
                k = k.replace('.lateral_conv0.', '.down_conv_5_4.')
                k = k.replace('.C3_p4.', '.down_csp_4_4.')
                k = k.replace('.reduce_conv1.', '.down_conv_4_3.')
                k = k.replace('.C3_p3.', '.down_csp_3_3.')
                k = k.replace('.bu_conv2.', '.up_conv_3_3.')
                k = k.replace('.C3_n3.', '.up_csp_3_4.')
                k = k.replace('.bu_conv1.', '.up_conv_4_4.')
                k = k.replace('.C3_n4.', '.up_csp_4_5.')
                k = k.replace('neck.', 'backbone.neck.')
                new_ckpt[k] = v
            return new_ckpt

    def inference_impl(
            self,
            batch: ImageInferenceType,
            buffer: Optional[YOLOXBuffer],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BBoxesInferenceType, Optional[Dict]]:
        images = torch.from_numpy(batch.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)
        features = self.backbone(images)

        if buffer is not None:
            past_indices = buffer['prev_indices']
            past_features = buffer['prev_features']
        else:
            past_indices = []
            past_features = []

        if past_time_constant is None:
            if len(past_features) > 0:
                ptc = [-1]
                pf = past_features[-1:]
            else:
                ptc = []
                pf = []
            # ptc = [p-default_interval for p in past_indices[-default_max_past:]]

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
            features = concat_pyramids([features]*len(ftc))
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=50, pad_value=0.0), new_buffer
    
    def forward_impl(
            self,
            batch: BatchTDict,
    ) -> Union[BatchTDict, LossDict]:
        if self.hparams.train_mask and self.training:
            # random mask past and future
            MAX_P = 3
            MAX_F = 1
            TP = batch['image']['clip_id'].size(1) - 1
            TF = batch['bbox']['clip_id'].size(1) - 1
            SAMPLE_P = min(TP, MAX_P)
            SAMPLE_F = min(TF, MAX_F)

            image_masks = random.sample(range(SAMPLE_P), k=random.randint(1, SAMPLE_P)) + [TP]
            image_masks.sort()
            for k in batch['image'].keys():
                batch['image'][k] = batch['image'][k][:, image_masks]

            bbox_masks = [0] + random.sample(range(1, SAMPLE_F+1), k=random.randint(1, SAMPLE_F))
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
            # neck
            if hasattr(v, 'p_attn') and getattr(v, 'p_attn').requires_grad is True:
                p_add.append(v.p_attn)
                logger.info(f'p_add: {k}')
            elif ('fc_out_list' in k or 'fc_query' in k or 'fc_key' in k) and hasattr(v, 'weight'):
                p_add.append(v.weight)
                if hasattr(v, 'bias'):
                    p_add.append(v.bias)
                logger.info(f'p_add: {k}')
            else:
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
        optimizer = torch.optim.SGD(
            [
                {'params': p_normal},
                {'params': p_wd, 'weight_decay': self.hparams.weight_decay},
                {'params': p_normal_lr, 'lr': self.hparams.lr * 5},
                {'params': p_wd_lr, 'lr': self.hparams.lr * 5, 'weight_decay': self.hparams.weight_decay},
                {'params': p_add, 'lr': self.hparams.lr * 50}
            ],
            lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True
        )
        if self.hparams.train_scheduler == 'yolox':
            scheduler = StreamYOLOScheduler(
                optimizer,
                int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
            )
        elif self.hparams.train_scheduler == 'msca':
            scheduler = MSCAScheduler(
                optimizer,
                int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
            )
        else:
            raise ValueError(f'scheduler {self.hparams.scheduler} not supported!')

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.hparams.neck_type == 'ta':
            p_attn = torch.cat([b.p_attn.flatten() for b in self.neck.blocks])
            self.log_dict({'p_mean': torch.mean(p_attn), 'p_std': torch.std(p_attn)}, on_step=True)
            attn_weight = torch.mean(torch.stack([b.attn_weight for b in self.neck.blocks], dim=0), dim=0)
            for f, w in enumerate(attn_weight.flip(0)):
                for p, ww in enumerate(w):
                    if f == p == 0:
                        self.log(f'w_-{p}_+{f}', ww, on_step=True, prog_bar=True)
                    else:
                        self.log(f'w_-{p}_+{f}', ww, on_step=True)
        elif self.hparams.neck_type == 'ta2':
            """
            attn_weight = self.neck.vis_attn_weight
            for f, w in enumerate(attn_weight.flip(0)):
                for p, ww in enumerate(w):
                    self.log(f'w_-{p}_+{f}', ww, on_step=True, prog_bar=True)
            """


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
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        neck_type: Literal['ta', 'ta2', 'ta3'] = 'ta',
        neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'relu',
        neck_p_init: Union[float, Literal['uniform', 'normal'], None] = 0.0,
        neck_tpe_merge: Literal['add', 'mul'] = 'add',
        neck_dropout: float = 0.0,
        train_mask: bool = False,
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> MSCASystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return MSCASystem(**__d)


def msca_m(
        data_source: Optional[BaseDataSource] = None,
        strategy: Optional[BaseSAPStrategy] = None,

        past_time_constant: List[int] = None,
        future_time_constant: List[int] = None,
        num_classes: int = 8,
        base_depth: int = 2,
        base_channel: int = 48,
        base_neck_depth: int = 3,
        hidden_ratio: float = 1.0,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (192, 384, 768),
        mid_channel: int = 192,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        neck_type: Literal['ta', 'ta2', 'ta3'] = 'ta',
        neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'relu',
        neck_p_init: Union[float, Literal['uniform', 'normal'], None] = 0.0,
        neck_tpe_merge: Literal['add', 'mul'] = 'add',
        neck_dropout: float = 0.0,
        train_mask: bool = False,
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.4,
        ignore_value: float = 1.7,
        **kwargs
) -> MSCASystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return MSCASystem(**__d)


def msca_l(
        data_source: Optional[BaseDataSource] = None,
        strategy: Optional[BaseSAPStrategy] = None,

        past_time_constant: List[int] = None,
        future_time_constant: List[int] = None,
        num_classes: int = 8,
        base_depth: int = 3,
        base_channel: int = 64,
        base_neck_depth: int = 3,
        hidden_ratio: float = 1.0,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (256, 512, 1024),
        mid_channel: int = 256,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        neck_type: Literal['ta', 'ta2', 'ta3'] = 'ta',
        neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'relu',
        neck_p_init: Union[float, Literal['uniform', 'normal'], None] = 0.0,
        neck_tpe_merge: Literal['add', 'mul'] = 'add',
        neck_dropout: float = 0.0,
        train_mask: bool = False,
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.5,
        ignore_value: float = 1.6,
        **kwargs
) -> MSCASystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return MSCASystem(**__d)
