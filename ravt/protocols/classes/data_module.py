import functools
from typing import Optional, Tuple, Callable, List, get_args

import pytorch_lightning as pl
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, default_collate

from ..structures import (
    BatchDict, DatasetConfigsRequiredKeys, ComponentDict,
    InternalConfigs, SubsetTypes, ComponentTypes, StageTypes
)
from .phase_init import PhaseInitMixin, ConfigTypes


class BaseDataset(PhaseInitMixin, pl.LightningDataModule):
    class WrapperClipDataset(Dataset):
        def __init__(
                self,
                dataset: 'BaseDataset',
                subset: SubsetTypes,
                required_keys: DatasetConfigsRequiredKeys,
        ):
            self.dataset = dataset
            self.subset = subset
            self.required_keys = required_keys

            self.items: List[Tuple[int, int]] = functools.reduce(list.__add__, [
                [
                    (seq_id, frame_id)
                    for frame_id in range(0, seq_len - self.required_keys['margin'], self.required_keys['interval'])
                ]
                for seq_id, seq_len in enumerate(self.dataset.get_lengths(self.subset))
            ])

        def __len__(self):
            return len(self.items)

        def __getitem__(self, item: int) -> BatchDict:
            seq_id, frame_id = self.items[item]
            batch: BatchDict = {}
            for key in get_args(ComponentTypes):  # type: ComponentTypes
                batch[key] = default_collate([
                    self.dataset.get_component(self.subset, key, seq_id, frame_id + i)
                    for i in self.required_keys['components'][key]
                ])
            return batch

    def __init__(
            self,
            batch_size: int,
            num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.eval_subset = None

        self.train_loader_factory: Optional[Callable[[], DataLoader]] = None
        self.val_loader_factory: Optional[Callable[[], DataLoader]] = None
        self.test_loader_factory: Optional[Callable[[], DataLoader]] = None

    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict:
        raise NotImplementedError()

    def get_lengths(self, subset: SubsetTypes) -> List[int]:
        """Get the lengths of sequences"""
        raise NotImplementedError()

    def get_coco(self, subset: SubsetTypes) -> COCO:
        """Optionally provide COCO"""
        raise NotImplementedError()

    def init_loader_factory(self, subset: SubsetTypes) -> Callable[[], DataLoader]:
        train = subset == 'train'
        dataset = BaseDataset.WrapperClipDataset(
            self, subset, self.hparams.required_keys_train if train else self.hparams.required_keys_eval
        )
        return functools.partial(
            DataLoader,
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=train,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'dataset':
            self.hparams['required_keys_train'] = configs['dataset']['required_keys_train']
            self.hparams['required_keys_eval'] = configs['dataset']['required_keys_eval']
            configs['evaluation']['coco_factory'] = self._evaluation_get_coco
        return configs

    def _evaluation_get_coco(self) -> Optional[COCO]:
        if self.eval_subset is None:
            return None
        try:
            return self.get_coco(self.eval_subset)
        except NotImplementedError:
            return None

    def setup(self, stage: StageTypes) -> None:
        if stage == 'fit' and self.train_loader_factory is None:
            self.train_loader_factory = self.init_loader_factory('train')
        if stage in ['fit', 'validate'] and self.val_loader_factory is None:
            self.val_loader_factory = self.init_loader_factory('eval')
            self.eval_subset = 'eval'
        if stage == 'test' and self.test_loader_factory is None:
            self.test_loader_factory = self.init_loader_factory('test')
            self.eval_subset = 'test'

    def train_dataloader(self):
        return self.train_loader_factory()

    def val_dataloader(self):
        return self.val_loader_factory()

    def test_dataloader(self):
        return self.test_loader_factory()
