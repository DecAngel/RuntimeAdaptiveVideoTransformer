import contextlib
import functools
import io
import itertools
from typing import List, Tuple

import pandas as pd
import cv2
from pycocotools.coco import COCO
import numpy as np

from ravt.core.constants import (
    ComponentNDict, SubsetLiteral, ComponentLiteral
)
from ravt.core.utils.array_operations import clip_or_pad_along
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.shared_memory import SharedMemoryClient
from ravt.core.base_classes import BaseDataSource
from ravt.core.configs import dataset_dir


img_dir = dataset_dir.joinpath('argoverse', 'Argoverse-1.1', 'tracking')
ann_dir = dataset_dir.joinpath('argoverse', 'Argoverse-HD', 'annotations')
dataset_configs = {
    'train': (img_dir, ann_dir.joinpath('train.json')),
    'eval': (img_dir, ann_dir.joinpath('val.json')),
    'test': (img_dir, ann_dir.joinpath('val.json')),
}


class ArgoverseDataSource(BaseDataSource):
    def __init__(
            self,
            subset: SubsetLiteral,
            size: Tuple[int, int] = (600, 960),
            max_objs: int = 100,
            enable_cache: bool = True,
    ):
        super().__init__()
        self.img_dir, self.ann_file = dataset_configs[subset]
        self.size = np.array(size, dtype=np.int32)
        self.max_objs = max_objs
        self.enable_cache = enable_cache
        self.default_coordinates = np.zeros((self.max_objs, 4), dtype=np.float32)
        self.default_labels = np.zeros((self.max_objs,), dtype=np.int32)

    def init(self):
        _ = self.image_frame
        _ = self.caches_image_bool_client

    @functools.cached_property
    def caches_image_bool_client(self):
        if self.enable_cache:
            c = SharedMemoryClient()
            if c.test_connection():
                B, C, H, W = sum(self.get_length()), 3, self.size[0], self.size[1]
                try:
                    cache_image = c.request_shared_memory(
                        f'argoverse_{self.ann_file.name}_image', dtype=np.uint8, shape_tuple=(B, C, H, W)
                    )
                    cache_bool = c.request_shared_memory(
                        f'argoverse_{self.ann_file.name}_bool', dtype=bool, shape_tuple=(B,)
                    )
                    return cache_image, cache_bool, c
                except RuntimeError:
                    logger.warning('Shm allocation failed, falling back to normal loading')
            else:
                logger.warning('Shm server connection failed, falling back to normal loading')
        else:
            logger.warning('Shm disabled!')
        return None

    @functools.cached_property
    def coco(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return COCO(str(self.ann_file))

    @functools.cached_property
    def class_frame(self):
        class_ids = sorted(self.coco.getCatIds())
        class_names = list(c['name'] for c in self.coco.loadCats(class_ids))
        return pd.DataFrame({
            'class_id': class_ids,
            'class_name': class_names,
        })

    @functools.cached_property
    def sequence_frame(self):
        seq_dirs = list(self.img_dir.joinpath(s) for s in self.coco.dataset['seq_dirs'])
        seq_lens = []
        first_img_id = None
        first_seq_id = None
        for i, image_id in enumerate(self.coco.getImgIds()):
            img = self.coco.loadImgs([image_id])[0]
            if i == 0:
                first_img_id = image_id
                first_seq_id = img['sid']
            assert i + first_img_id == image_id, 'img_id not contiguous'
            if img['sid'] == len(seq_lens) - 1 + first_seq_id:
                # continuous seq
                assert img['fid'] == seq_lens[-1], 'fid not contiguous'
                seq_lens[-1] += 1
            else:
                # new seq
                assert img['sid'] == len(seq_lens) + first_seq_id, 'sid not contiguous'
                assert img['fid'] == 0, 'fid not starting from 0'
                seq_lens.append(1)
        seq_start_img_ids = list(itertools.accumulate([first_img_id] + seq_lens[:-1]))
        return pd.DataFrame({
            'seq_len': seq_lens,
            'seq_dir': seq_dirs,
            'seq_start_img_id': seq_start_img_ids,
        }, index=range(first_seq_id, first_seq_id + len(seq_lens)))

    @functools.cached_property
    def image_frame(self):
        image_ids = []
        image_paths = []
        image_sizes = []
        gt_coordinates = []
        gt_labels = []
        class_ids = self.class_frame['class_id'].tolist()
        for seq_len, seq_dir, seq_start_img_id in self.sequence_frame.itertuples(index=False):
            for image_id in range(seq_start_img_id, seq_start_img_id + seq_len):
                image_ids.append(image_id)
                image_ann = self.coco.loadImgs([image_id])[0]
                image_sizes.append(np.array([image_ann['height'], image_ann['width']], dtype=np.int32))
                image_paths.append(str(seq_dir.joinpath(image_ann['name']).resolve()))

                label_ids = self.coco.getAnnIds(imgIds=[image_id])
                label_ann = self.coco.loadAnns(label_ids)
                if len(label_ann) > 0:
                    bbox = np.array([l['bbox'] for l in label_ann], dtype=np.float32)
                    cls = np.array([class_ids.index(l['category_id']) for l in label_ann], dtype=np.int32)
                    bbox[:, 2:] += bbox[:, :2]

                    # clip bbox & filter small bbox
                    bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], a_min=0, a_max=image_ann['width'])
                    bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], a_min=0, a_max=image_ann['height'])
                    mask = np.min(bbox[:, 2:] - bbox[:, :2], axis=1) >= 2
                    if not np.all(mask):
                        logger.warning(f'filtered {bbox[np.logical_not(mask)]}!')

                    bbox = bbox[mask]
                    cls = cls[mask]
                else:
                    bbox = np.zeros((0, 4), dtype=np.float32)
                    cls = np.zeros((0,), dtype=np.int32)
                gt_coordinates.append(bbox)
                gt_labels.append(cls)
        return pd.DataFrame({
            'image_path': image_paths,
            'image_size': image_sizes,
            'gt_coordinate': pd.Series(gt_coordinates, dtype=object, index=image_ids),
            'gt_label': pd.Series(gt_labels, dtype=object, index=image_ids),
        }, index=image_ids)

    def _get_image(self, image_id: int) -> np.ndarray:
        image_path, _, _, _ = self.image_frame.loc[image_id, :]
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.size[::-1])
        image = image.transpose(2, 0, 1)
        return image

    def get_component(self, seq_id: int, frame_id: int, component: ComponentLiteral) -> ComponentNDict:
        image_id = self.sequence_frame.iloc[seq_id, 2] + frame_id
        if component == 'meta':
            return {
                'seq_id': np.array(seq_id, dtype=np.int32),
                'frame_id': np.array(frame_id, dtype=np.int32),
                'image_id': np.array(image_id, dtype=np.int32)
            }
        elif component == 'image':
            if self.caches_image_bool_client is not None:
                index_id = image_id - self.image_frame.index[0]
                if self.caches_image_bool_client[1][index_id].item() is False:
                    image = self._get_image(image_id)
                    self.caches_image_bool_client[0][index_id] = image
                    self.caches_image_bool_client[1][index_id] = True
                else:
                    image = self.caches_image_bool_client[0][index_id]
            else:
                image = self._get_image(image_id)
            original_size = self.image_frame.loc[image_id, 'image_size']
            return {
                'image': image,
                'original_size': original_size,
            }
        elif component == 'bbox':
            _, original_size, coordinate, label = self.image_frame.loc[image_id, :]
            resize_ratio = (self.size / original_size).astype(np.float32)[[1, 0, 1, 0]]
            coordinate = coordinate * resize_ratio
            original_size = self.image_frame.loc[image_id, 'image_size']
            return {
                'coordinate': clip_or_pad_along(coordinate, 0, self.max_objs),
                'label': clip_or_pad_along(label, 0, self.max_objs),
                'probability': np.ones((self.max_objs, ), dtype=np.float32),
                'current_size': self.size,
                'original_size': original_size,
            }
        else:
            raise ValueError(f'Unsupported component {component}!')

    def get_length(self) -> List[int]:
        return self.sequence_frame['seq_len'].tolist()
