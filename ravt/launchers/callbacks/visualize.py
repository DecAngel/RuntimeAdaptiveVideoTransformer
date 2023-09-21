from typing import Optional, Tuple, Literal

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from ravt.core.utils.phase_init import PhaseInitMixin
from ravt.core.constants import PredDict, BatchDict, PhaseTypes, AllConfigs


class VisualizeCallback(PhaseInitMixin, pl.Callback):
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
        self.stage = None

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'visualization':
            self.visualize_dir = configs['environment']['output_visualize_dir']
            self.stage = configs['internal']['stage']
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

        all_ids = sorted(list(set(gt_image_ids + gt_bbox_ids + pred_bbox_ids)))

        vis_images = []
        gt_images = batch['image']['image'][0]
        if self.resize is not None:
            original_size = gt_images.shape[-2:]
            resize_ratio = np.array([self.resize[0]/original_size[0], self.resize[1]/original_size[1]])
            resize_ratio = resize_ratio[[1, 0, 1, 0]]
            with torch.inference_mode():
                gt_images = F.interpolate(gt_images, size=self.resize)
        else:
            resize_ratio = np.ones((4, ), dtype=np.float)

        gt_images = gt_images.cpu().permute(0, 2, 3, 1).numpy()
        blank_image = np.zeros_like(gt_images[0])

        for i in all_ids:
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
                mask = ((overlay[..., 1] == 0) * 255).astype(np.uint8)
                image = np.bitwise_and(image, mask[..., None])
                image = cv2.add(image, overlay)

            if i in pred_bbox_ids:
                overlay = blank_image.copy()
                for c, l, p in zip(
                    pred['bbox']['coordinate'][0, pred_bbox_ids.index(i)].cpu().numpy(),
                    pred['bbox']['label'][0, pred_bbox_ids.index(i)].cpu().numpy(),
                    pred['bbox']['probability'][0, pred_bbox_ids.index(i)].cpu().numpy(),
                ):
                    if np.sum(c) > 1e-1:
                        overlay = cv2.rectangle(
                            overlay, *((c * resize_ratio).astype(int).reshape(2, 2).tolist()),
                            (0, 0, 255), 2
                        )
                    else:
                        break
                mask = ((overlay[..., 2] == 0) * 255).astype(np.uint8)
                image = np.bitwise_and(image, mask[..., None])
                image = cv2.add(image, overlay)

            image = np.pad(image, ((50, 10), (5, 5), (0, 0)), constant_values=(0, 0))
            image = cv2.putText(
                image, f'Frame {i-all_ids[0]}', (0, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2.5, color=(255, 255, 255), thickness=2
            )
            vis_images.append(image)

        return np.concatenate(vis_images, axis=1)

    def show(self, batch: BatchDict, pred: PredDict) -> None:
        cv2.imshow('visualize', self.visualize(batch, pred))
        cv2.waitKey(1)

    def save(self, batch: BatchDict, pred: PredDict) -> None:
        vis_image = self.visualize(batch, pred)
        seq_id = batch['image']['seq_id'][0, 0].cpu().numpy().item()
        frame_id = batch['image']['frame_id'][0, 0].cpu().numpy().item()
        cv2.imwrite(str(self.visualize_dir.joinpath(f'{self.stage}_{seq_id:03d}_{frame_id:03d}.jpg')), vis_image)

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
            self.execute(batch, outputs)
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
            self.execute(batch, outputs)
            if self.visualize_count >= self.visualize_test:
                self.visualize_count = 0
