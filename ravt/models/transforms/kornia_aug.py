from typing import List, Optional

import torch
import kornia.augmentation as ka

from ravt.core.constants import BatchDict, AllConfigs, PhaseTypes, ComponentTypes
from ravt.core.base_classes import BaseTransform


def listify(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    else:
        return [x]


class KorniaAugmentation(BaseTransform):
    # TODO: support image bbox different length
    def __init__(
            self,
            train_aug: Optional[ka.VideoSequential] = None,
            train_resize: Optional[ka.VideoSequential] = None,
            eval_aug: Optional[ka.VideoSequential] = None,
            eval_resize: Optional[ka.VideoSequential] = None,
    ):
        super().__init__()
        self.train_aug = train_aug
        self.train_resize = train_resize
        self.eval_aug = eval_aug
        self.eval_resize = eval_resize
        self.train_aug_transform = None
        self.train_resize_transform = None
        self.eval_aug_transform = None
        self.eval_resize_transform = None
        self.input_keys_train: Optional[List[ComponentTypes]] = None
        self.input_keys_eval: Optional[List[ComponentTypes]] = None
        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'model':
            key_mapping = {'image': 'image', 'bbox': 'bbox'}
            required_keys_train = list(configs['internal']['required_keys_train'].keys())
            required_keys_eval = list(configs['internal']['required_keys_eval'].keys())
            self.input_keys_train = [k for k in key_mapping if k in required_keys_train]
            self.train_aug_transform = ka.AugmentationSequential(
                self.train_aug,
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_train],
                same_on_batch=False,
                keepdim=False,
            ) if self.train_aug is not None else None
            self.train_resize_transform = ka.AugmentationSequential(
                self.train_resize,
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_train],
                same_on_batch=True,
                keepdim=False,
            ) if self.train_resize is not None else None
            self.input_keys_eval = [k for k in key_mapping if k in required_keys_eval]
            self.eval_aug_transform = ka.AugmentationSequential(
                self.eval_aug,
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_eval],
                same_on_batch=False,
                keepdim=False,
            ) if self.eval_aug is not None else None
            self.eval_resize_transform = ka.AugmentationSequential(
                self.eval_resize,
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_eval],
                same_on_batch=True,
                keepdim=False,
            ) if self.eval_resize is not None else None
        return configs

    def forward(self, batch: BatchDict) -> BatchDict:
        if self.training:
            def transform(*args):
                args = listify(self.train_aug_transform(*args)) if self.train_aug_transform is not None else args
                args = listify(self.train_resize_transform(*args)) if self.train_resize_transform is not None else args
                return args
            input_keys = self.input_keys_train
        else:
            def transform(*args):
                args = listify(self.eval_aug_transform(*args)) if self.eval_aug_transform is not None else args
                args = listify(self.eval_resize_transform(*args)) if self.eval_resize_transform is not None else args
                return args
            input_keys = self.input_keys_eval

        inputs = []
        if 'image' in input_keys:
            inputs.append(batch['image']['image'] / self.c255)
        if 'bbox' in input_keys:
            coordinates = batch['bbox']['coordinate']
            b, t, o, c = coordinates.shape
            coordinates = coordinates[..., [0, 1, 2, 1, 2, 3, 0, 3]].reshape(b, t, o, 4, 2)
            inputs.append(coordinates)

        outputs = transform(*inputs)

        if 'image' in input_keys:
            batch['image']['image'] = (outputs[0] * self.c255).type(torch.uint8)
            outputs.pop(0)
        if 'bbox' in input_keys:
            xy_min, xy_max = torch.aminmax(outputs[0], dim=-2)
            coordinates = torch.cat([xy_min, xy_max], dim=-1)
            batch['bbox']['coordinate'] = coordinates
            outputs.pop(0)

        return batch
