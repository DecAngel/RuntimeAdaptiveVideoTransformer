import contextlib
from typing import Optional, Dict, Tuple, Union, List, TypedDict, Sequence, Literal

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import nn

from ravt.data_samplers import YOLOXDataSampler
from ravt.metrics import COCOEvalMAPMetric
from ravt.core.constants import (
    BatchTDict, LossDict, SubsetLiteral
)
from ravt.core.base_classes import BaseSystem, BaseDataSampler, BaseMetric, BaseTransform, BaseDataSource, \
    BaseSAPStrategy
from ravt.core.utils.visualization import draw_feature, draw_bbox, add_clip_id, draw_feature_batch, draw_image, \
    draw_flow, draw_grid_clip_id
from ravt.core.utils.collection_operations import tensor2ndarray, reverse_collate, select_collate
from ravt.core.utils.grad_check import plot_grad_flow
from .blocks import StreamYOLOScheduler, MSCAScheduler
from .blocks.types import PYRAMID

from .yolox_base import YOLOXBaseSystem, YOLOXBuffer, concat_pyramids
from .blocks.backbones import YOLOXPAFPNBackbone
from .blocks.necks import DeformNeck, DeformNeck
from .blocks.heads import TALHead
from ...core.utils.array_operations import clip_or_pad_along


class WarpStreamNetSystem(YOLOXBaseSystem):
    def __init__(
            self,
            data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
            strategy: Optional[BaseSAPStrategy] = None,
            batch_size: int = 1,
            num_workers: int = 0,

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
        self.save_hyperparameters(ignore=['kwargs', 'data_sources', 'strategy'])

        super().__init__(
            backbone=YOLOXPAFPNBackbone(**self.hparams),
            neck=DeformNeck(**self.hparams),
            head=TALHead(**self.hparams),
            batch_size=batch_size,
            num_workers=num_workers,
            with_bbox_0_train=True,
            data_sources=data_sources,
            data_sampler=YOLOXDataSampler(
                1, [-self.hparams.predict_num, 0], [self.hparams.predict_num],
                [[-i, 0] for i in range(1, self.hparams.predict_num + 1)],
                [[0, i] for i in range(1, self.hparams.predict_num + 1)],
            ),
            metric=COCOEvalMAPMetric(future_time_constant=[predict_num]),
            strategy=strategy,
        )

    def visualize(
            self,
            batch: BatchTDict,
            features_p: PYRAMID,
            features_f: PYRAMID,
            pred: Optional[BatchTDict],
    ):
        seq_id = batch['seq_id'][0].cpu().numpy().item()
        frame_id = batch['frame_id'][0].cpu().numpy().item()
        TP = features_p[0].size(1) - 1
        TF = features_f[0].size(1)

        f = tensor2ndarray({'feature': torch.cat([features_p[0], features_f[0]], dim=1)})  # B, TP+1+TF, C, H, W
        f = select_collate(f, 0)  # TP+1+TF, C, H, W

        flows = tensor2ndarray({'flow': self.neck.vis_flows})
        flows = select_collate(flows, 0)

        clip_ids = (
                batch['image']['clip_id'][0].cpu().numpy().astype(int).tolist() +
                batch['bbox']['clip_id'][0,
                (int(self.with_bbox_0_train) if self.training else 0):].cpu().numpy().astype(int).tolist()
        )
        vis_feature = draw_feature_batch(reverse_collate(f), size=(300, 480))
        vis_image = [
            draw_image(self.active_data_source.get_component(seq_id, frame_id + c, 'image'), size=(300, 480))
            for c in clip_ids
        ]
        vis_flow = [draw_flow(ff, size=(300, 480)) for ff in reverse_collate(flows)]
        vis_flow.extend([np.zeros_like(vis_flow[0])] * (TP + 1 + TF - len(vis_flow)))

        vis = draw_grid_clip_id([vis_image, vis_feature, vis_flow], clip_ids)
        self.image_writer.write(vis)

        self.time_recorder.print()

    def inference_impl(
            self,
            batch: BatchTDict,
            buffer: Optional[YOLOXBuffer],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BatchTDict, Optional[Dict]]:
        # TODO: change
        # Ignore ptc and ftc
        images = torch.from_numpy(batch.astype(np.float32)).permute(2, 0, 1)[None, None, ...].to(device=self.device)
        features = self.backbone(images)

        # Buffer and neck
        new_buffer = {
            'prev_features': [features],
        }
        if buffer is None or 'prev_features' not in buffer:
            features = self.neck(concat_pyramids([features, features]))
        else:
            features = self.neck(concat_pyramids([*buffer['prev_features'], features]))
        pred_dict = self.head(features, shape=images.shape[-2:])

        return clip_or_pad_along(np.concatenate([
            pred_dict['pred_coordinates'].cpu().numpy()[0],
            pred_dict['pred_probabilities'].cpu().numpy()[0, :, :, None],
            pred_dict['pred_labels'].cpu().numpy()[0, :, :, None].astype(float),
        ], axis=2), axis=1, fixed_length=100, pad_value=0.0), new_buffer

    def forward_impl(
            self,
            batch: BatchTDict,
    ) -> Union[BatchTDict, LossDict]:
        images: Float[torch.Tensor, 'B TP0 C H W'] = batch['image']['image'].float()
        coordinates: Optional[Float[torch.Tensor, 'B TF0 O C']] = batch['bbox'][
            'coordinate'] if 'bbox' in batch else None
        labels: Optional[Int[torch.Tensor, 'B TF0 O']] = batch['bbox']['label'] if 'bbox' in batch else None
        past_frame_constant = batch['image']['clip_id'][:, :-1].float()
        future_frame_constant = batch['bbox']['clip_id'][:,
                                (int(self.with_bbox_0_train) if self.training else 0):].float()

        features_p = self.backbone(images)
        features_f, loss_dict = self.neck(features_p, past_frame_constant, future_frame_constant)

        if self.training:
            hamming_loss = loss_dict['hamming_loss']
            smooth_loss = loss_dict['smooth_loss']
            head_loss = self.head(
                features_f,
                gt_coordinates=coordinates,
                gt_labels=labels,
                shape=images.shape[-2:]
            )['loss']
            res = {
                'loss': (1 / (self.fraction_epoch ** 2 + 1)) * hamming_loss +
                        (0.5 / (self.fraction_epoch ** 2 + 1)) * smooth_loss +
                        head_loss,
                'hamming_loss': hamming_loss,
                'smooth_loss': smooth_loss,
                'head_loss': head_loss,
            }
            vis_pred = None
        else:
            pred_dict = self.head(features_f, shape=images.shape[-2:])
            res = {
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
            vis_pred = res

        if self.image_writer is not None and not self.trainer.sanity_checking and self.active_data_source is not None:
            self.visualization_count += 1
            if self.visualization_count >= self.trainer.log_every_n_steps:
                self.visualization_count = 0
                self.visualize(batch, features_p, features_f, vis_pred)

        return res

    def configure_optimizers(self):
        p_bias, p_norm, p_weight = [], [], []
        all_parameters = []
        all_parameters.extend(self.backbone.named_modules())
        all_parameters.extend(self.neck.named_modules())
        all_parameters.extend(self.head.named_modules())

        for k, v in all_parameters:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                p_bias.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                p_norm.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                p_weight.append(v.weight)  # apply decay

        default_options = {
            'lr': self.hparams.lr,
            'momentum': self.hparams.momentum,
            'weight_decay': 0.0,
            'nesterov': True,
        }
        params = [
            {'params': p_norm},
            {'params': p_weight, 'weight_decay': self.hparams.weight_decay},
            {'params': p_bias},
        ]
        params = list(filter(lambda p: len(p['params']) > 0, params))
        optimizer = torch.optim.SGD(params=params, **default_options)

        scheduler = StreamYOLOScheduler(
            optimizer,
            int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]


def warp_streamnet_s(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
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
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        **kwargs
) -> WarpStreamNetSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return WarpStreamNetSystem(**__d)


def warp_streamnet_m(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 1,
        num_classes: int = 8,
        base_depth: int = 2,
        base_channel: int = 48,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (192, 384, 768),
        mid_channel: int = 192,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.4,
        ignore_value: float = 1.7,
        **kwargs
) -> WarpStreamNetSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return WarpStreamNetSystem(**__d)


def warp_streamnet_l(
        data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
        strategy: Optional[BaseSAPStrategy] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        predict_num: int = 1,
        num_classes: int = 8,
        base_depth: int = 3,
        base_channel: int = 64,
        strides: Tuple[int, ...] = (8, 16, 32),
        in_channels: Tuple[int, ...] = (256, 512, 1024),
        mid_channel: int = 256,
        depthwise: bool = False,
        act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',
        max_objs: int = 100,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ignore_thr: float = 0.5,
        ignore_value: float = 1.6,
        **kwargs
) -> WarpStreamNetSystem:
    __d = locals().copy()
    __d.update(kwargs)
    del __d['kwargs']
    return WarpStreamNetSystem(**__d)
