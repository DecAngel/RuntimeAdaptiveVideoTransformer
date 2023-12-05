import uuid
from typing import Optional, Dict, Protocol, List, Tuple, Iterator

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler

from ..utils.lightning_logger import ravt_logger as logger
from ..constants import SubsetLiteral, ComponentLiteral
from .data_source import BaseDataSource
from .data_sampler import BaseDataSampler


class DataSampler(Sampler[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]):
    def __init__(self, clips: List[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]):
        super().__init__(clips)
        self.clips = clips

    def __iter__(self) -> Iterator[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]:
        return iter(self.clips)

    def __len__(self):
        return len(self.clips)


class DataModuleMixin(Protocol):
    data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]]
    data_sampler: Optional[BaseDataSampler]
    batch_size: int
    num_workers: int

    def worker_init_fn(self, worker_id: int) -> None:
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        seed = uuid.uuid4().int % 2 ** 32
        pl.seed_everything(seed)
        logger.info(f'Initializing worker {worker_id} with seed {seed}')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_sources['train'],
            sampler=DataSampler(self.data_sampler.sample(True, self.data_sources['train'].get_length())),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            worker_init_fn=self.worker_init_fn if self.num_workers != 0 else None,
            persistent_workers=self.num_workers != 0,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_sources['eval'],
            sampler=DataSampler(self.data_sampler.sample(False, self.data_sources['eval'].get_length())),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            worker_init_fn=self.worker_init_fn if self.num_workers != 0 else None,
            persistent_workers=self.num_workers != 0,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_sources['test'],
            sampler=DataSampler(self.data_sampler.sample(False, self.data_sources['test'].get_length())),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            worker_init_fn=self.worker_init_fn if self.num_workers != 0 else None,
            persistent_workers=self.num_workers != 0,
        )
