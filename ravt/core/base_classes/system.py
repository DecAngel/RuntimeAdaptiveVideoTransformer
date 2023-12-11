from typing import Optional, Dict

import pytorch_lightning as pl

from ..utils.image_writer import ImageWriter

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # low pytorch version where LRScheduler is protected
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..constants import SubsetLiteral
from .inference_mixin import InferenceMixin
from .checkpoint_mixin import CheckpointMixin
from .data_module_mixin import DataModuleMixin
from .data_source import BaseDataSource
from .data_sampler import BaseDataSampler
from .transform import BaseTransform
from .metric import BaseMetric
from .sap_strategy import BaseSAPStrategy


class BaseSystem(DataModuleMixin, CheckpointMixin, InferenceMixin, pl.LightningModule):
    def __init__(
            self,
            data_sources: Optional[Dict[SubsetLiteral, BaseDataSource]] = None,
            data_sampler: Optional[BaseDataSampler] = None,
            batch_size: int = 1,
            num_workers: int = 0,
            transform: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
            strategy: Optional[BaseSAPStrategy] = None,
    ):
        super().__init__()
        self.data_sources = data_sources
        self.data_sampler = data_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.metric = metric
        self.strategy = strategy
        self.image_writer: Optional[ImageWriter] = None
