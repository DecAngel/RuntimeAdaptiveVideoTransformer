import pytorch_lightning as pl
from torch.utils.data import Dataset, default_collate, DataLoader

from ..constants import BatchDict, SubsetTypes, ComponentTypes
from ..utils.lightning_logger import ravt_logger as logger
from ..base_classes import BaseDataSource, BaseDataSampler


class WrapperClipDataset(Dataset):
    def __init__(self, data_source: BaseDataSource, sampler: BaseDataSampler, subset: SubsetTypes):
        super().__init__()
        self.data_source = data_source
        self.subset = subset
        self.samples = sampler.sample(subset, data_source.get_length(subset))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int) -> BatchDict:
        sample_dict = self.samples[item]
        seq_id = sample_dict['seq_id']
        frame_id = sample_dict['frame_id']
        image_clip_ids = sample_dict['image']
        bbox_clip_ids = sample_dict['bbox']
        batch: BatchDict = {
            'image': default_collate([
                {**self.data_source.get_component(self.subset, 'image', seq_id, frame_id + i), 'clip_id': i}
                for i in image_clip_ids
            ]),
            'bbox': default_collate([
                {**self.data_source.get_component(self.subset, 'bbox', seq_id, frame_id + i), 'clip_id': i}
                for i in bbox_clip_ids
            ]),
        }

        return batch


class LocalDataModule(pl.LightningDataModule):
    def __init__(self, data_source: BaseDataSource, sampler: BaseDataSampler, batch_size: int, num_workers: int):
        super().__init__()
        self.data_source = data_source
        self.sampler = sampler
        self.save_hyperparameters(ignore=['data_source', 'sampler'])

        def worker_init_fn(worker_id: int) -> None:
            import torch.multiprocessing
            torch.multiprocessing.set_sharing_strategy('file_system')
            logger.debug(f'Initializing worker {worker_id}')

        self.worker_init_fn = worker_init_fn

    def train_dataloader(self):
        return DataLoader(
            WrapperClipDataset(self.data_source, self.sampler, 'train'),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            worker_init_fn=self.worker_init_fn if self.hparams.num_workers != 0 else None,
            shuffle=True,
            persistent_workers=self.hparams.num_workers != 0,
        )

    def val_dataloader(self):
        return DataLoader(
            WrapperClipDataset(self.data_source, self.sampler, 'eval'),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            worker_init_fn=self.worker_init_fn if self.hparams.num_workers != 0 else None,
            shuffle=False,
            persistent_workers=self.hparams.num_workers != 0,
        )

    def test_dataloader(self):
        return DataLoader(
            WrapperClipDataset(self.data_source, self.sampler, 'test'),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            worker_init_fn=self.worker_init_fn if self.hparams.num_workers != 0 else None,
            shuffle=False,
            persistent_workers=self.hparams.num_workers != 0,
        )
