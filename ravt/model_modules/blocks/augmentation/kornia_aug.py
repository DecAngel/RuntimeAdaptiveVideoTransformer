import random
from typing import List, Optional

import torch.nn as nn
import kornia.augmentation as ka

from ravt.protocols.structures import BatchDict, ConfigTypes, InternalConfigs, ComponentTypes
from ravt.protocols.classes import BaseTransform


class KorniaSequential(nn.ModuleList):
    def __init__(self, *modules: nn.Module):
        super().__init__(modules)

    def forward(self, *args):
        for m in self.modules():
            args = m(*args)
        return args


class KorniaRandomChoice(nn.ModuleList):
    def __init__(self, *modules: nn.Module):
        super().__init__(modules)

    def forward(self, *args):
        return random.choice(list(self.modules()))(*args)


class KorniaIdentity(nn.Module):
    def forward(self, *args):
        return args


class KorniaAugmentation(BaseTransform):
    def __init__(
            self,
            train_aug: Optional[nn.Module],
            train_resize: Optional[nn.Module],
            eval_aug: Optional[nn.Module],
            eval_resize: Optional[nn.Module],
    ):
        super().__init__()
        self.train_aug = train_aug or KorniaIdentity()
        self.train_resize = train_resize or KorniaIdentity()
        self.eval_aug = eval_aug or KorniaIdentity()
        self.eval_resize = eval_resize or KorniaIdentity()
        self.train_aug_transform = None
        self.train_resize_transform = None
        self.eval_aug_transform = None
        self.eval_resize_transform = None
        self.input_keys_train: Optional[List[ComponentTypes]] = None
        self.input_keys_eval: Optional[List[ComponentTypes]] = None

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'model':
            key_mapping = {'image': 'image', 'bbox': 'bbox_xyxy'}
            required_keys_train = list(configs['dataset']['required_keys_train']['components'].keys())
            required_keys_eval = list(configs['dataset']['required_keys_eval']['components'].keys())
            self.input_keys_train = [k for k in key_mapping if k in required_keys_train]
            self.train_aug_transform = ka.AugmentationSequential(
                ka.VideoSequential(self.train_aug),
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_train],
                same_on_batch=False,
                keepdim=False,
            )
            self.train_aug_transform = ka.AugmentationSequential(
                ka.VideoSequential(self.train_resize),
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_train],
                same_on_batch=True,
                keepdim=False,
            )
            self.input_keys_eval = [k for k in key_mapping if k in required_keys_eval]
            self.eval_aug_transform = ka.AugmentationSequential(
                ka.VideoSequential(self.eval_aug),
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_eval],
                same_on_batch=False,
                keepdim=False,
            )
            self.eval_resize_transform = ka.AugmentationSequential(
                ka.VideoSequential(self.eval_resize),
                data_keys=[key_mapping[k] for k in key_mapping if k in required_keys_eval],
                same_on_batch=True,
                keepdim=False,
            )
        return configs

    def forward(self, batch: BatchDict) -> BatchDict:
        inputs = []
        if 'image' in self.input_keys:
            inputs.append(batch['image']['image'])
        if 'bbox' in self.input_keys:
            inputs.append(batch['bbox']['coordinate'])

        if self.training:
            outputs = self.train_resize_transform(*self.train_aug_transform(*inputs))
        else:
            outputs = self.eval_resize_transform(*self.eval_aug_transform(*inputs))

        if len(inputs) == 1:
            outputs = [outputs]
        if 'image' in self.input_keys:
            original_size = batch['image']['image'].shape[-2:]
            current_size = outputs[0].shape[-2:]
            batch['image']['image'] = outputs[0]
            batch['image']['resize_ratio'][..., 0] *= original_size[0]/current_size[0]
            batch['image']['resize_ratio'][..., 1] *= original_size[1]/current_size[1]
            outputs.pop(0)
        if 'bbox' in self.input_keys:
            batch['bbox']['coordinate'] = outputs[0]
            outputs.pop(0)

        return batch
