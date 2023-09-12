from typing import Optional, Tuple, Literal

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from ravt.protocols.classes import PhaseInitMixin
from ravt.protocols.structures import PredDict, BatchDict, ConfigTypes, InternalConfigs


class VisualizeCallback(PhaseInitMixin, pl.Callback):
    # TODO: save visualize file in sub directory
    def __init__(
            self,
            resize: Optional[Tuple[int, int]] = None,
            mode: Literal['show', 'save', 'tensorboard'] = 'save',
            visualize_train_interval: int = 0,
            visualize_eval_interval: int = 0,
            visualize_test_interval: int = 0,
    ):
        super().__init__()
        self.resize = resize
        self.mode = mode
        self.visualize_train = visualize_train_interval
        self.visualize_eval = visualize_eval_interval
        self.visualize_test = visualize_test_interval

        self._trainer: Optional[pl.Trainer] = None
        self.visualize_dir = None
        self.visualize_count = 0
        self.save_count = 0

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'summary':
            self.visualize_dir = configs['environment']['output_visualize_dir']
            self.save_count = len(list(self.visualize_dir.iterdir()))
        return configs

    def visualize(self, batch: BatchDict, pred: PredDict) -> np.ndarray:
        if 'image' in batch:
            gt_image_ids = batch['image']['image_id'][0].cpu().numpy().tolist()
        else:
            raise ValueError('\'image\' must be present in batch!')
        if 'bbox' in batch:
            gt_bbox_ids = batch['bbox']['image_id'][0].cpu().numpy().tolist()
        else:
            gt_bbox_ids = []

        if 'bbox' in pred:
            pred_bbox_ids = pred['bbox']['image_id'][0].cpu().numpy().tolist()
        else:
            pred_bbox_ids = []

        all_ids = gt_image_ids + gt_bbox_ids + pred_bbox_ids
        from_id = min(all_ids)
        to_id = max(all_ids) + 1

        vis_images = []
        gt_images = batch['image']['image'][0]
        if self.resize is not None:
            original_size = gt_images.shape[-2:]
            resize_ratio = np.array([self.resize[0]/original_size[0], self.resize[1]/original_size[1]])
            resize_ratio = resize_ratio[[1, 0, 1, 0]]
            gt_images = F.interpolate(gt_images, size=self.resize)
        else:
            resize_ratio = np.ones((4, ), dtype=np.float)

        gt_images = (gt_images*255).cpu().to(dtype=torch.uint8).permute(0, 2, 3, 1)[..., [2, 1, 0]].numpy()
        blank_image = np.zeros_like(gt_images[0])

        for i in range(from_id, to_id):
            if i in gt_image_ids:
                image = gt_images[gt_image_ids.index(i)]
            else:
                image = blank_image.copy()

            if i in gt_bbox_ids:
                overlay = blank_image.copy()
                for c, l in zip(
                    batch['bbox']['coordinate'][0, gt_bbox_ids.index(i)].cpu().numpy(),
                    batch['bbox']['label'][0, gt_bbox_ids.index(i)].cpu().numpy(),
                ):
                    if np.sum(c) > 1e-5:
                        overlay = cv2.rectangle(overlay, *((c * resize_ratio).astype(int).reshape(2, 2).tolist()), (0, 255, 0), 2)
                    else:
                        break
                image = cv2.add(image, overlay)

            if i in pred_bbox_ids:
                overlay = blank_image.copy()
                for c, l, p in zip(
                    pred['bbox']['coordinate'][0, pred_bbox_ids.index(i)].cpu().numpy(),
                    pred['bbox']['label'][0, pred_bbox_ids.index(i)].cpu().numpy(),
                    pred['bbox']['probability'][0, pred_bbox_ids.index(i)].cpu().numpy(),
                ):
                    if np.sum(c) > 1e-5:
                        overlay = cv2.rectangle(overlay, *((c * resize_ratio).astype(int).reshape(2, 2).tolist()), (0, 0, int(255*p.item())), 2)
                    else:
                        break
                image = cv2.add(image, overlay)

            vis_images.append(image)
        return np.concatenate(vis_images, axis=1)

    def show(self, batch: BatchDict, pred: PredDict) -> None:
        cv2.imshow('visualize', self.visualize(batch, pred))

    def save(self, batch: BatchDict, pred: PredDict) -> None:
        vis_image = self.visualize(batch, pred)
        cv2.imwrite(str(self.visualize_dir.joinpath(f'{self.save_count:05}.jpg')), vis_image)
        self.save_count += 1

    def tensorboard(self, batch: BatchDict, pred: PredDict) -> None:
        raise NotImplementedError()

    def execute(self, batch: BatchDict, pred: PredDict) -> None:
        if self.mode == 'show':
            self.show(batch, pred)
        elif self.mode == 'save':
            self.save(batch, pred)
        elif self.mode == 'tensorboard':
            self.tensorboard(batch, pred)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')
        self.visualize_count += 1

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._trainer = trainer
        self.visualize_count = 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: PredDict, batch: BatchDict, batch_idx: int
    ) -> None:
        if self.visualize_train != 0:
            self.execute(batch, {})
            if self.visualize_count >= self.visualize_train:
                self.visualize_count = 0

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: PredDict,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.visualize_eval != 0 and not self._trainer.sanity_checking:
            self.save(batch, outputs)
            if self.visualize_count >= self.visualize_eval:
                self.visualize_count = 0

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: PredDict,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.visualize_test != 0 and not self._trainer.sanity_checking:
            self.save(batch, outputs)
            if self.visualize_count >= self.visualize_test:
                self.visualize_count = 0
