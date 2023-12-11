from typing import List, Optional, Tuple

import torch
import kornia.augmentation as ka

from ravt.core.constants import BatchTDict, ComponentLiteral
from ravt.core.base_classes import BaseTransform


def listify(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    else:
        return [x]


class KorniaAugmentation(BaseTransform):
    def __init__(
            self,
            train_component_types: Tuple[ComponentLiteral] = ('image', 'bbox'),
            train_aug: Optional[ka.VideoSequential] = None,
            train_resize: Optional[ka.VideoSequential] = None,
            eval_component_types: Tuple[ComponentLiteral] = ('image',),
            eval_aug: Optional[ka.VideoSequential] = None,
            eval_resize: Optional[ka.VideoSequential] = None,
    ):
        super().__init__()
        key_mapping = {'image': 'image', 'bbox': 'bbox'}
        self.input_keys_train = [k for k in key_mapping if k in train_component_types]
        self.train_aug_transform = ka.AugmentationSequential(
            train_aug,
            data_keys=[key_mapping[k] for k in key_mapping if k in train_component_types],
            same_on_batch=False,
            keepdim=False,
        ) if train_aug is not None else None
        self.train_resize_transform = ka.AugmentationSequential(
            train_resize,
            data_keys=[key_mapping[k] for k in key_mapping if k in train_component_types],
            same_on_batch=True,
            keepdim=False,
        ) if train_resize is not None else None
        self.input_keys_eval = [k for k in key_mapping if k in eval_component_types]
        self.eval_aug_transform = ka.AugmentationSequential(
            eval_aug,
            data_keys=[key_mapping[k] for k in key_mapping if k in eval_component_types],
            same_on_batch=False,
            keepdim=False,
        ) if eval_aug is not None else None
        self.eval_resize_transform = ka.AugmentationSequential(
            eval_resize,
            data_keys=[key_mapping[k] for k in key_mapping if k in eval_component_types],
            same_on_batch=True,
            keepdim=False,
        ) if eval_resize is not None else None

        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)

    def preprocess_tensor(self, batch: BatchTDict) -> BatchTDict:
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

        max_time = max([i.size(1) for i in inputs])
        difference_time = [max_time - i.size(1) for i in inputs]
        for i, d in enumerate(difference_time):
            if d > 0:
                inputs[i] = torch.cat([inputs[i], inputs[i][:, [0]*d]], dim=1)
        outputs = list(transform(*inputs))
        for i, d in enumerate(difference_time):
            if d > 0:
                outputs[i] = outputs[i][:, :-d]

        if 'image' in input_keys:
            batch['image']['image'] = (outputs[0] * self.c255).type(torch.uint8)
            outputs.pop(0)
        if 'bbox' in input_keys:
            xy_min, xy_max = torch.aminmax(outputs[0], dim=-2)
            coordinates = torch.cat([xy_min, xy_max], dim=-1)
            batch['bbox']['coordinate'] = coordinates
            outputs.pop(0)

        return batch
