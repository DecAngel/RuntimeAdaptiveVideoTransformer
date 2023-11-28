from typing import Optional, Tuple, Literal, List

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import flow_to_image

from ravt.core.base_classes import BaseDataSource
from ravt.core.base_classes.system import BaseSystem
from ravt.core.constants import PredDict, BatchDict, SubsetTypes, VisualizationDict
from ravt.core.configs import output_visualize_dir


class VisualizeCallback(pl.Callback):
    def __init__(
            self,
            exp_tag: str,
            resize: Tuple[int, int] = (600, 960),
            display_mode: Literal['show_image', 'save_image', 'save_video', 'tensorboard_image'] = 'save_video',
            batch_mode: Literal['all', 'first'] = 'all',
            visualize_train_interval: int = 0,
            visualize_eval_interval: int = 0,
            visualize_test_interval: int = 0,
    ):
        super().__init__()
        self.exp_tag = exp_tag
        self.resize = resize
        self.display_mode = display_mode
        self.batch_mode = batch_mode
        self.visualize_train = visualize_train_interval
        self.visualize_eval = visualize_eval_interval
        self.visualize_test = visualize_test_interval

        self._trainer: Optional[pl.Trainer] = None
        self._logger: Optional[SummaryWriter] = None
        self._data_source: Optional[BaseDataSource] = None

        self.video_writer: Optional[cv2.VideoWriter] = None

        self.visualize_dir = output_visualize_dir.joinpath(f'{exp_tag}')
        self.visualize_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_count = 0
        self.stage = None

    def setup(self, trainer: "pl.Trainer", pl_module: BaseSystem, stage: str) -> None:
        self._trainer = trainer
        self._data_source = pl_module.data_source
        if pl_module.logger is not None:
            self._logger = pl_module.logger.experiment
        self.visualize_count = 0

    def _get_visualize_image_and_ids(
            self, batch: BatchDict, pred: VisualizationDict, batch_id: int
    ) -> Tuple[np.ndarray, int, int]:
        if 'image' in batch:
            base_seq_id = batch['image']['seq_id'][batch_id, -1].cpu().numpy().item()
            base_frame_id = batch['image']['frame_id'][batch_id, -1].cpu().numpy().item()
        else:
            raise ValueError('\'image\' must be present in batch!')
        if 'bbox' in pred:
            pred_bbox_clip_ids = pred['bbox']['clip_id'][batch_id].cpu().numpy().tolist()
        else:
            pred_bbox_clip_ids = []
        if 'visualization' in pred:
            pass

        vis_images = []
        for i in pred_bbox_clip_ids:
            # image
            image_dict = self._data_source.get_component(self.stage, 'image', base_seq_id, base_frame_id + i)
            image = image_dict['image'].transpose(1, 2, 0)
            if self.resize is not None:
                original_size = image.shape[:2]
                resize_ratio = np.array([self.resize[0] / original_size[0], self.resize[1] / original_size[1]])
                resize_ratio = resize_ratio[[1, 0, 1, 0]]
                image = cv2.resize(image, dsize=(self.resize[1], self.resize[0]), interpolation=cv2.INTER_LINEAR)
            else:
                resize_ratio = np.ones((4,), dtype=np.float32)
            blank_image = np.zeros_like(image)

            # gt_bbox
            overlay = blank_image.copy()
            bbox_dict = self._data_source.get_component(self.stage, 'bbox', base_seq_id, base_frame_id + i)
            for c, l in zip(
                    bbox_dict['coordinate'],
                    bbox_dict['label'],
            ):
                if np.sum(c) > 1e-1:
                    overlay = cv2.rectangle(
                        overlay, *((c * resize_ratio).astype(int).reshape(2, 2).tolist()),
                        (0, 255, 0), 2
                    )
                else:
                    break
            mask = ((overlay[..., 1] == 0) * 255).astype(np.uint8)
            image = np.bitwise_and(image, mask[..., None])
            image = cv2.add(image, overlay)

            # pred_bbox
            overlay = blank_image.copy()
            for c, l, p in zip(
                    pred['bbox']['coordinate'][batch_id, pred_bbox_clip_ids.index(i)].cpu().numpy(),
                    pred['bbox']['label'][batch_id, pred_bbox_clip_ids.index(i)].cpu().numpy(),
                    pred['bbox']['probability'][batch_id, pred_bbox_clip_ids.index(i)].cpu().numpy(),
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
                image, f'Frame {base_frame_id}', (0, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2.5, color=(255, 255, 255), thickness=2
            )
            vis_images.append(image)
        return np.concatenate(vis_images, axis=1), base_seq_id, base_frame_id

    def get_batch_visualize_image_and_ids(self, batch: BatchDict, pred: PredDict):
        B = batch['image']['image'].size(0)
        if self.batch_mode == 'all':
            return [self._get_visualize_image_and_ids(batch, pred, i) for i in range(B)]
        elif self.batch_mode == 'first':
            return [self._get_visualize_image_and_ids(batch, pred, i) for i in [0]]

    def on_start(self, stage: SubsetTypes):
        self.stage = stage

    def on_step(self, batch: BatchDict, pred: PredDict):
        vis_images = self.get_batch_visualize_image_and_ids(batch, pred)

        if self.display_mode == 'show_image':
            for i, (v, seq_id, frame_id) in enumerate(vis_images):
                cv2.imshow(f'visualize_{i}', v)
            cv2.waitKey(1)
        elif self.display_mode == 'save_image':
            for v, seq_id, frame_id in vis_images:
                cv2.imwrite(str(self.visualize_dir.joinpath(f'{self.stage}_{seq_id:03d}_{frame_id:03d}.jpg')), v)
        elif self.display_mode == 'save_video':
            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(
                    str(self.visualize_dir.joinpath(f'{self.stage}_{self._trainer.current_epoch}.mp4')),
                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                    30,
                    (vis_images[0][0].shape[1], vis_images[0][0].shape[0])
                )
            for v, seq_id, frame_id in vis_images:
                self.video_writer.write(v)
        elif self.display_mode == 'tensorboard_image':
            for i, (v, seq_id, frame_id) in enumerate(vis_images):
                self._logger.add_image(f'{self.stage}_vis', v, global_step=self._trainer.global_step*len(vis_images)+i)
        else:
            raise ValueError(f'Unsupported mode {self.display_mode}')
        self.visualize_count += 1

    def on_end(self):
        self.stage = None
        if self.display_mode == 'save_video' and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_start('train')

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_start('eval')

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_start('test')

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: PredDict, batch: BatchDict, batch_idx: int
    ) -> None:
        if self.visualize_train != 0:
            if self.visualize_count % self.visualize_train == 0:
                self.on_step(batch, {})

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
            if self.visualize_count % self.visualize_eval == 0:
                self.on_step(batch, outputs)

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
            if self.visualize_count % self.visualize_eval == 0:
                self.on_step(batch, outputs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_end()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_end()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_end()


"""
def get_visualize_image(self, batch: BatchDict, pred: PredDict, batch_id: int) -> np.ndarray:
    if 'image' in batch:
        gt_image_ids = batch['image']['clip_id'][batch_id].cpu().numpy().tolist()
    else:
        raise ValueError('\'image\' must be present in batch!')
    if 'bbox' in batch:
        gt_bbox_ids = batch['bbox']['image_id'][batch_id].cpu().numpy().tolist()
    else:
        gt_bbox_ids = []
    if 'bbox' in pred:
        pred_bbox_ids = pred['bbox']['image_id'][batch_id].cpu().numpy().tolist()
    else:
        pred_bbox_ids = []

    all_ids = sorted(list(set(gt_image_ids + gt_bbox_ids + pred_bbox_ids)))

    vis_images = []
    gt_images = batch['image']['image'][batch_id]
    if self.resize is not None:
        original_size = gt_images.shape[-2:]
        resize_ratio = np.array([self.resize[0]/original_size[0], self.resize[1]/original_size[1]])
        resize_ratio = resize_ratio[[1, 0, 1, 0]]
        with torch.inference_mode():
            gt_images = F.interpolate(gt_images, size=self.resize)
    else:
        resize_ratio = np.ones((4, ), dtype=np.float32)

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
                batch['bbox']['coordinate'][batch_id, gt_bbox_ids.index(i)].cpu().numpy(),
                batch['bbox']['label'][batch_id, gt_bbox_ids.index(i)].cpu().numpy(),
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
                pred['bbox']['coordinate'][batch_id, pred_bbox_ids.index(i)].cpu().numpy(),
                pred['bbox']['label'][batch_id, pred_bbox_ids.index(i)].cpu().numpy(),
                pred['bbox']['probability'][batch_id, pred_bbox_ids.index(i)].cpu().numpy(),
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
"""