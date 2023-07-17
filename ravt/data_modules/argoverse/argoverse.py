import contextlib
import io
import itertools
from pathlib import Path
from typing import Union, Optional, TypedDict, List

import typeguard
import cv2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import pytorch_lightning as pl
import imgaug.augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
from jaxtyping import Float, Int, UInt

from ravt.common.array_operations import clip_or_pad_along
from ravt.configs import dataset_dir


class ArgoverseDataset(pl.LightningDataModule):
    class OutputTypedDict(TypedDict):
        seq_ids: int
        frame_ids: int
        image_ids: int
        resize_ratios: Float[np.ndarray, 'hw=2']
        images: List[UInt[np.ndarray, 'height width channels_rgb=3']]
        gt_coordinates: Optional[List[Float[np.ndarray, 'max_objs coords_xyxy=4']]]
        gt_labels: Optional[List[Int[np.ndarray, 'max_objs']]]

    class ArgoverseSubset(Dataset):
        def __init__(
                self,
                dataset_dir: Path,
                annotation_file: str,
                transform: Optional[iaa.Augmenter] = None,
                clip_image_length: int = 4,
                clip_label_length: int = 4,
                max_objs: int = 100,
                visualize: bool = False
        ):
            super().__init__()
            self.img_dir = Path(dataset_dir).joinpath('Argoverse-1.1', 'tracking')
            self.ann_dir = Path(dataset_dir).joinpath('Argoverse-HD', 'annotations')
            with contextlib.redirect_stdout(io.StringIO()):
                self.coco = COCO(str(self.ann_dir.joinpath(f'{annotation_file}.json')))
            self.seq_dirs = list(self.img_dir.joinpath(s) for s in self.coco.dataset['seq_dirs'])
            self.class_ids = sorted(self.coco.getCatIds())
            self.class_names = list(c['name'] for c in self.coco.loadCats(self.class_ids))
            self.transform = transform or iaa.Identity()
            self.clip_image_length = clip_image_length
            self.clip_label_length = clip_label_length
            self.max_objs = max_objs
            self.visualize = visualize

            # check sequence lengths and assert img_id = frame_id + prev_seq_lens
            self.seq_lens = []
            self.first_img_id = None
            self.first_seq_id = None
            for i, img_id in enumerate(self.coco.getImgIds()):
                img = self.coco.loadImgs([img_id])[0]
                if i == 0:
                    self.first_img_id = img_id
                    self.first_seq_id = img['sid']

                assert i + self.first_img_id == img_id, 'img_id not contiguous'

                if img['sid'] == len(self.seq_lens) - 1 + self.first_seq_id:
                    # continuous seq
                    assert img['fid'] == self.seq_lens[-1], 'fid not contiguous'
                    self.seq_lens[-1] += 1
                else:
                    # new seq
                    assert img['sid'] == len(self.seq_lens) + self.first_seq_id, 'sid not contiguous'
                    assert img['fid'] == 0, 'fid not starting from 0'
                    self.seq_lens.append(1)

            self.seq_start_img_id = list(itertools.accumulate([self.first_img_id] + self.seq_lens[:-1]))

            # create clips
            self.clips = []
            for seq_id in range(len(self.seq_lens)):
                for frame_id in range(self.clip_image_length-1, self.seq_lens[seq_id] - self.clip_label_length + 1):
                    self.clips.append((seq_id, frame_id, self.seq_start_img_id[seq_id]+frame_id))

            self.default_coordinates = np.zeros((self.max_objs, 4), dtype=np.float32)
            self.default_labels = np.zeros((self.max_objs, ), dtype=np.int32)

        def __len__(self):
            return len(self.clips)

        @typeguard.typechecked()
        def __getitem__(
                self, clip_id: int
        ) -> 'ArgoverseDataset.OutputTypedDict':
            """Get a clip according to clip_id.

            :param clip_id:
            :return:
            """
            seq_id, frame_id, center_image_id = self.clips[clip_id]

            origin_hw: Optional[np.ndarray] = None
            resize_ratio: Optional[np.ndarray] = None

            self.transform.seed_()
            transform: iaa.Augmenter = self.transform.to_deterministic()

            images = []
            for i in range(1-self.clip_image_length, 1):
                image_name = self.coco.loadImgs([center_image_id+i])[0]['name']
                image = cv2.imread(str(self.seq_dirs[seq_id].joinpath(image_name)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                augmented_image = transform.augment_image(image)
                if origin_hw is None and resize_ratio is None:
                    origin_hw = np.array(image.shape[:2], dtype=np.int32)
                    augmented_hw = np.array(augmented_image.shape[:2], dtype=np.float32)
                    resize_ratio = origin_hw / augmented_hw
                images.append(augmented_image)

            gt_coordinates = []
            gt_labels = []
            for i in range(0, self.clip_label_length):
                label_ids = self.coco.getAnnIds(imgIds=[center_image_id+i])
                label_ann = self.coco.loadAnns(label_ids)
                bbox = np.array([l['bbox'] for l in label_ann], dtype=np.float32)
                cls = np.array([self.class_ids.index(l['category_id']) for l in label_ann], dtype=np.int32)

                coordinates = self.default_coordinates
                labels = self.default_labels
                if bbox.ndim == 2:
                    bbox[:, 2:] += bbox[:, :2]

                    label_bbox = transform.augment_bounding_boxes(BoundingBoxesOnImage(
                        [BoundingBox(*b, c) for b, c in zip(bbox, cls)],
                        shape=(*origin_hw, )
                    ))
                    label_bbox.remove_out_of_image_fraction_(0.5)
                    label_bbox.clip_out_of_image_()

                    if len(label_bbox) != 0:
                        coordinates = clip_or_pad_along(
                            np.stack([b.coords.flatten() for b in label_bbox], axis=0),
                            axis=0,
                            fixed_length=self.max_objs
                        )
                        labels = clip_or_pad_along(
                            np.array([b.label for b in label_bbox], dtype=np.int32),
                            axis=0,
                            fixed_length=self.max_objs
                        )

                gt_coordinates.append(coordinates)
                gt_labels.append(labels)

            return {
                "seq_ids": seq_id,
                "frame_ids": frame_id,
                "images": images,
                "resize_ratios": resize_ratio,
                "gt_coordinates": gt_coordinates,
                "gt_labels": gt_labels,
                "image_ids": center_image_id,
            }

    def __init__(
            self,
            dataset_argoverse_dir: Union[Path, str] = dataset_dir.joinpath('argoverse'),
            train_transform: Optional[iaa.Augmenter] = None,
            val_transform: Optional[iaa.Augmenter] = None,
            clip_image_length: int = 4,
            clip_label_length: int = 4,
            max_objs: int = 100,
            batch_size: int = 8,
            num_workers: int = 8,
            visualize: bool = False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore='kwargs')
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: str) -> None:
        if stage == 'fit' and self.train_set is None:
            self.train_set = ArgoverseDataset.ArgoverseSubset(
                self.hparams.dataset_argoverse_dir,
                'train_1',
                self.hparams.train_transform,
                self.hparams.clip_image_length,
                self.hparams.clip_label_length,
                self.hparams.max_objs,
                self.hparams.visualize,
            )
        if stage in ['fit', 'validate'] and self.val_set is None:
            self.val_set = ArgoverseDataset.ArgoverseSubset(
                self.hparams.dataset_argoverse_dir,
                'train_2',
                self.hparams.val_transform,
                self.hparams.clip_image_length,
                self.hparams.clip_label_length,
                self.hparams.max_objs,
                self.hparams.visualize,
            )
        if stage == 'test' and self.test_set is None:
            self.test_set = ArgoverseDataset.ArgoverseSubset(
                self.hparams.dataset_argoverse_dir,
                'val',
                self.hparams.val_transform,
                self.hparams.clip_image_length,
                self.hparams.clip_label_length,
                self.hparams.max_objs,
                self.hparams.visualize,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
