import contextlib
import functools
import io
import itertools
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
import cv2
from pycocotools.coco import COCO
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp

from ravt.protocols.structures import (
    BBoxComponentDict, ImageComponentDict, MetaComponentDict, ConfigTypes, InternalConfigs, ComponentTypes,
    ComponentDict, SubsetTypes
)
from ravt.protocols.classes import BaseDataset
from ravt.utils.array_operations import clip_or_pad_along


def torch_mp_cached_component(fn):
    manager = mp.Manager()
    cache = manager.dict()

    @functools.wraps(fn)
    def wrapper(self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int):
        nonlocal manager, cache
        key = (seq_id, frame_id)
        if key not in cache:
            cache[key] = manager.dict(fn(self, subset, component, seq_id, frame_id))
        return {**cache[key]}

    def clear():
        nonlocal cache
        cache.clear()

    wrapper.clear = clear
    return wrapper


class ArgoverseDataset(BaseDataset):
    class Subset:
        def __init__(self, dataset_dir: Path, subset_file: str, size: Tuple[int, int], max_objs: int):
            img_dir = dataset_dir.joinpath('argoverse', 'Argoverse-1.1', 'tracking')
            ann_dir = dataset_dir.joinpath('argoverse', 'Argoverse-HD', 'annotations')
            self.coco_path = str(ann_dir.joinpath(subset_file))
            self.size = size
            self.max_objs = max_objs

            coco = self.get_coco()

            # class
            class_ids = sorted(coco.getCatIds())
            class_names = list(c['name'] for c in coco.loadCats(class_ids))
            self.class_frame = pd.DataFrame({
                'class_id': class_ids,
                'class_name': class_names
            })

            # sequence
            # check sequence lengths and assert img_id = frame_id + prev_seq_lens
            seq_dirs = list(img_dir.joinpath(s) for s in coco.dataset['seq_dirs'])
            self.seq_lens = []
            first_img_id = None
            first_seq_id = None
            for i, image_id in enumerate(coco.getImgIds()):
                img = coco.loadImgs([image_id])[0]
                if i == 0:
                    first_img_id = image_id
                    first_seq_id = img['sid']
                assert i + first_img_id == image_id, 'img_id not contiguous'
                if img['sid'] == len(self.seq_lens) - 1 + first_seq_id:
                    # continuous seq
                    assert img['fid'] == self.seq_lens[-1], 'fid not contiguous'
                    self.seq_lens[-1] += 1
                else:
                    # new seq
                    assert img['sid'] == len(self.seq_lens) + first_seq_id, 'sid not contiguous'
                    assert img['fid'] == 0, 'fid not starting from 0'
                    self.seq_lens.append(1)
            seq_start_img_ids = list(itertools.accumulate([first_img_id] + self.seq_lens[:-1]))
            self.sequence_frame = pd.DataFrame({
                'seq_len': self.seq_lens,
                'seq_dir': seq_dirs,
                'seq_start_img_id': seq_start_img_ids,
            }, index=range(first_seq_id, first_seq_id + len(self.seq_lens)))

            # images
            image_ids = []
            image_paths = []
            image_sizes = []
            gt_coordinates = []
            gt_labels = []
            for seq_len, seq_dir, seq_start_img_id in self.sequence_frame.itertuples(index=False):
                for image_id in range(seq_start_img_id, seq_start_img_id + seq_len):
                    image_ids.append(image_id)
                    image_ann = coco.loadImgs([image_id])[0]
                    image_sizes.append((image_ann['height'], image_ann['width']))
                    image_paths.append(str(seq_dir.joinpath(image_ann['name']).resolve()))

                    label_ids = coco.getAnnIds(imgIds=[image_id])
                    label_ann = coco.loadAnns(label_ids)
                    if len(label_ann) > 0:
                        bbox = np.array([l['bbox'] for l in label_ann], dtype=np.float32)
                        cls = np.array([class_ids.index(l['category_id']) for l in label_ann], dtype=np.int32)
                        bbox[:, 2:] += bbox[:, :2]
                    else:
                        bbox = np.zeros((0, 4), dtype=np.float32)
                        cls = np.zeros((0,), dtype=np.int32)
                    gt_coordinates.append(bbox)
                    gt_labels.append(cls)
            self.image_frame = pd.DataFrame({
                'image_path': image_paths,
                'image_size': image_sizes,
                'gt_coordinate': pd.Series(gt_coordinates, dtype=object, index=image_ids),
                'gt_label': pd.Series(gt_labels, dtype=object, index=image_ids),
            }, index=image_ids)

            original_size = self.image_frame.iloc[0, 1]
            resize_ratio = np.array(
                [original_size[0]/self.size[0], original_size[1]/self.size[1]],
                dtype=np.float32
            )
            self.original_size = torch.tensor(original_size, dtype=torch.int)
            self.resize_ratio = resize_ratio[[1, 0, 1, 0]]

            # defaults
            self.default_coordinates = np.zeros((self.max_objs, 4), dtype=np.float32)
            self.default_labels = np.zeros((self.max_objs,), dtype=np.int32)

        def get_coco(self):
            with contextlib.redirect_stdout(io.StringIO()):
                return COCO(self.coco_path)

        def get_meta(self, seq_id: int, frame_id: int) -> MetaComponentDict:
            return {
                'seq_id': torch.tensor(seq_id, dtype=torch.int32),
                'frame_id': torch.tensor(frame_id, dtype=torch.int32),
                'image_id': torch.tensor(self.sequence_frame.iloc[seq_id, 2] + frame_id, dtype=torch.int32)
            }

        def get_image(self, seq_id: int, frame_id: int) -> ImageComponentDict:
            image_id = self.sequence_frame.iloc[seq_id, 2] + frame_id
            image_path, original_size, _, _ = self.image_frame.loc[image_id, :]
            image = cv2.imread(image_path)

            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)[None, [2, 1, 0], ...] / 255.0
            image = F.interpolate(image, self.size, mode='bilinear')[0]

            return {**self.get_meta(seq_id, frame_id)} | {
                'image': image,
                'original_size': self.original_size,
            }

        def get_bbox(self, seq_id: int, frame_id: int) -> BBoxComponentDict:
            image_id = self.sequence_frame.iloc[seq_id, 2] + frame_id
            _, _, coordinate, label = self.image_frame.loc[image_id, :]
            coordinate /= self.resize_ratio

            return {**self.get_meta(seq_id, frame_id)} | {
                'coordinate': clip_or_pad_along(torch.from_numpy(coordinate), 0, self.max_objs),
                'label': clip_or_pad_along(torch.from_numpy(label), 0, self.max_objs),
            }

        def get_lengths(self) -> List[int]:
            return self.seq_lens

    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            size: Tuple[int, int] = (600, 960),
            max_objs: int = 100,
    ):
        """Dataset instances should avoid storing native list or dict!!!"""
        super().__init__(batch_size, num_workers)
        self.save_hyperparameters()

    @functools.cache
    def subset(self, subset_type: SubsetTypes):
        if subset_type == 'train':
            return ArgoverseDataset.Subset(
                self.hparams.dataset_dir,
                'train_1.json',
                self.hparams.size,
                self.hparams.max_objs
            )
        elif subset_type == 'eval':
            return ArgoverseDataset.Subset(
                self.hparams.dataset_dir,
                'train_2.json',
                self.hparams.size,
                self.hparams.max_objs
            )
        elif subset_type == 'test':
            return ArgoverseDataset.Subset(
                self.hparams.dataset_dir,
                'val.json',
                self.hparams.size,
                self.hparams.max_objs
            )
        else:
            raise ValueError(f'Unsupported subset {subset_type}')

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'dataset':
            self.hparams['dataset_dir'] = configs['environment']['dataset_dir']
        return configs

    def get_coco(self, subset: SubsetTypes) -> COCO:
        return self.subset(subset).get_coco()

    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict:
        if component == 'image':
            return self.subset(subset).get_image(seq_id, frame_id)
        elif component == 'bbox':
            return self.subset(subset).get_bbox(seq_id, frame_id)
        else:
            raise ValueError(f'Unsupported component {component}')

    def get_lengths(self, subset: SubsetTypes) -> List[int]:
        return self.subset(subset).get_lengths()
