import functools
from typing import List, Tuple

import pytorch_lightning as pl
from torch.utils.data import Dataset, default_collate, DataLoader

from ..constants import BatchKeys, BatchDict, PhaseTypes, AllConfigs, SubsetTypes
from ..utils.phase_init import PhaseInitMixin
from ..base_classes import BaseDataSource


class WrapperClipDataset(Dataset):
    def __init__(
            self,
            data_source: BaseDataSource,
            subset: SubsetTypes,
            required_keys: BatchKeys,
    ):
        super().__init__()
        self.data_source = data_source
        self.subset = subset
        self.required_keys = required_keys

        self.items: List[Tuple[int, int]] = functools.reduce(list.__add__, [
            [
                (seq_id, frame_id)
                for frame_id in range(0, seq_len - self.required_keys['margin'], self.required_keys['interval'])
            ]
            for seq_id, seq_len in enumerate(self.data_source.get_length(self.subset))
        ])
        self.required_keys = {k: v for k, v in self.required_keys.items() if k not in ['margin', 'interval']}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item: int) -> BatchDict:
        seq_id, frame_id = self.items[item]
        batch: BatchDict = {}
        for key, value in self.required_keys.items():
            batch[key] = default_collate([
                self.data_source.get_component(self.subset, key, seq_id, frame_id + i)
                for i in value
            ])
        return batch


class LocalDataModule(PhaseInitMixin, pl.LightningDataModule):
    def __init__(self, data_source: BaseDataSource, batch_size: int, num_workers: int):
        super().__init__()
        self.data_source = data_source
        self.save_hyperparameters(ignore=['data_source'])

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'dataset':
            self.hparams.required_keys_train = configs['internal']['required_keys_train']
            self.hparams.required_keys_eval = configs['internal']['required_keys_eval']
        return configs

    def train_dataloader(self):
        return DataLoader(
            WrapperClipDataset(
                self.data_source,
                'train',
                self.hparams.required_keys_train,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=self.hparams.num_workers != 0,
        )

    def val_dataloader(self):
        return DataLoader(
            WrapperClipDataset(
                self.data_source,
                'eval',
                self.hparams.required_keys_eval,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.num_workers != 0,
        )

    def test_dataloader(self):
        return DataLoader(
            WrapperClipDataset(
                self.data_source,
                'test',
                self.hparams.required_keys_eval,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.num_workers != 0,
        )
