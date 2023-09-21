import contextlib
import functools
import io
import itertools
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import cv2
from pycocotools.coco import COCO
import numpy as np

from ravt.core.constants import (
    BBoxComponentDict, ImageComponentDict, ComponentDict, AllConfigs,
    SubsetTypes, PhaseTypes, ComponentTypes
)
from ravt.core.utils.array_operations import clip_or_pad_along
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.base_classes import BaseDataSource
from ravt.core.functional_classes import SharedMemoryClient


class ArgoverseDataSource(BaseDataSource):
    class Subset:
        def __init__(
                self, img_dir: Path, ann_file: Path, size: Tuple[int, int], max_objs: int,
                sm_client: Optional[SharedMemoryClient] = None,
        ):
            self.coco_path = str(ann_file)
            self.size = size
            self.max_objs = max_objs
            self.sm_client = sm_client

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
            self.first_img_id = None
            self.first_seq_id = None
            for i, image_id in enumerate(coco.getImgIds()):
                img = coco.loadImgs([image_id])[0]
                if i == 0:
                    self.first_img_id = image_id
                    self.first_seq_id = img['sid']
                assert i + self.first_img_id == image_id, 'img_id not contiguous'
                if img['sid'] == len(self.seq_lens) - 1 + self.first_seq_id:
                    # continuous seq
                    assert img['fid'] == self.seq_lens[-1], 'fid not contiguous'
                    self.seq_lens[-1] += 1
                else:
                    # new seq
                    assert img['sid'] == len(self.seq_lens) + self.first_seq_id, 'sid not contiguous'
                    assert img['fid'] == 0, 'fid not starting from 0'
                    self.seq_lens.append(1)
            seq_start_img_ids = list(itertools.accumulate([self.first_img_id] + self.seq_lens[:-1]))
            self.sequence_frame = pd.DataFrame({
                'seq_len': self.seq_lens,
                'seq_dir': seq_dirs,
                'seq_start_img_id': seq_start_img_ids,
            }, index=range(self.first_seq_id, self.first_seq_id + len(self.seq_lens)))

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
            self.original_size = np.array(original_size, dtype=np.int32)
            self.resize_ratio = resize_ratio[[1, 0, 1, 0]]

            # defaults
            self.default_coordinates = np.zeros((self.max_objs, 4), dtype=np.float32)
            self.default_labels = np.zeros((self.max_objs,), dtype=np.int32)

            if self.sm_client is not None:
                if self.sm_client.test_connection():
                    B, C, H, W = sum(self.get_length()), 3, self.size[0], self.size[1]
                    self.cache = self.sm_client.request_shared_memory(
                        f'argoverse_{ann_file.name}_image', dtype=np.uint8, shape_tuple=(B, C, H, W)
                    )
                    self.cache_bool = self.sm_client.request_shared_memory(
                        f'argoverse_{ann_file.name}_bool', dtype=bool, shape_tuple=(B, )
                    )
                else:
                    raise ConnectionRefusedError(f'Cannot connect to port {self.sm_client.port}')

        def get_coco(self):
            with contextlib.redirect_stdout(io.StringIO()):
                return COCO(self.coco_path)

        def get_meta(self, seq_id: int, frame_id: int):
            return {
                'seq_id': np.array(seq_id, dtype=np.int32),
                'frame_id': np.array(frame_id, dtype=np.int32),
                'image_id': np.array(self.sequence_frame.iloc[seq_id, 2] + frame_id, dtype=np.int32)
            }

        def _get_image(self, image_id: int) -> np.ndarray:
            image_path, original_size, _, _ = self.image_frame.loc[image_id, :]
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.size[::-1])
            image = image.transpose(2, 0, 1)
            return image

        def get_image(self, seq_id: int, frame_id: int) -> ImageComponentDict:
            image_id = self.sequence_frame.iloc[seq_id, 2] + frame_id
            if self.sm_client is not None:
                index_id = image_id - self.first_img_id
                if self.cache_bool[index_id].item() is False:
                    image = self._get_image(image_id)
                    self.cache[index_id] = image
                    self.cache_bool[index_id] = True
                else:
                    image = self.cache[index_id]
            else:
                image = self._get_image(image_id)

            return self.get_meta(seq_id, frame_id) | {
                'image': image,
                'original_size': self.original_size,
            }

        def get_bbox(self, seq_id: int, frame_id: int) -> BBoxComponentDict:
            image_id = self.sequence_frame.iloc[seq_id, 2] + frame_id
            _, _, coordinate, label = self.image_frame.loc[image_id, :]
            coordinate = coordinate / self.resize_ratio

            return self.get_meta(seq_id, frame_id) | {
                'coordinate': clip_or_pad_along(coordinate, 0, self.max_objs),
                'label': clip_or_pad_along(label, 0, self.max_objs),
            }

        def get_length(self) -> List[int]:
            return self.seq_lens

    def __init__(
            self,
            size: Tuple[int, int] = (600, 960),
            max_objs: int = 100,
            enable_cache: bool = True,
    ):
        """Dataset instances should avoid storing native list or dict!!!"""
        super().__init__()
        self.size = size
        self.max_objs = max_objs
        self.client = SharedMemoryClient() if enable_cache else None
        self.dataset_dir: Optional[Path] = None

    @functools.cached_property
    def img_dir(self):
        return self.dataset_dir.joinpath('argoverse', 'Argoverse-1.1', 'tracking')

    @functools.cached_property
    def ann_dir(self):
        return self.dataset_dir.joinpath('argoverse', 'Argoverse-HD', 'annotations')

    @functools.cached_property
    def ann_train_file(self):
        return self.ann_dir.joinpath('train_1.json')

    @functools.cached_property
    def ann_eval_file(self):
        return self.ann_dir.joinpath('train_2.json')

    @functools.cached_property
    def ann_test_file(self):
        return self.ann_dir.joinpath('val.json')

    @functools.cache
    def subset(self, subset_type: SubsetTypes) -> 'ArgoverseDataSource.Subset':
        if subset_type == 'train':
            return ArgoverseDataSource.Subset(
                self.img_dir,
                self.ann_train_file,
                self.size,
                self.max_objs,
                self.client,
            )
        elif subset_type == 'eval':
            return ArgoverseDataSource.Subset(
                self.img_dir,
                self.ann_eval_file,
                self.size,
                self.max_objs,
                # self.client,
            )
        elif subset_type == 'test':
            return ArgoverseDataSource.Subset(
                self.img_dir,
                self.ann_test_file,
                self.size,
                self.max_objs,
                # self.client,
            )
        else:
            raise ValueError(f'Unsupported subset {subset_type}')

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'dataset':
            self.dataset_dir = configs['environment']['dataset_dir']
            configs['extra']['train_coco_file'] = self.ann_train_file
            configs['extra']['eval_coco_file'] = self.ann_eval_file
            configs['extra']['test_coco_file'] = self.ann_test_file
        return configs

    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict:
        if component == 'image':
            return self.subset(subset).get_image(seq_id, frame_id)
        elif component == 'bbox':
            return self.subset(subset).get_bbox(seq_id, frame_id)
        else:
            raise ValueError(f'Unsupported component {component}')

    def get_length(self, subset: SubsetTypes) -> List[int]:
        return self.subset(subset).get_length()
