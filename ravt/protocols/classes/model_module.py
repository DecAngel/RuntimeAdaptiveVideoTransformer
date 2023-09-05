import contextlib
from typing import Union, Tuple, List

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from ..structures import DatasetConfigsRequiredKeys, BatchDict, LossDict, MetricDict
from .phase_init import PhaseInitMixin


class BaseTransform(PhaseInitMixin, nn.Module):
    def forward(self, batch: BatchDict) -> BatchDict:
        raise NotImplementedError()


class BaseMetric(PhaseInitMixin, Metric):
    def update(self, batch: BatchDict, pred: BatchDict) -> None:
        raise NotImplementedError()

    def compute(self) -> MetricDict:
        raise NotImplementedError()


class BaseModel(PhaseInitMixin, pl.LightningModule):
    def __init__(
            self,
            metric: BaseMetric
    ):
        super().__init__()
        self.metric = metric

    @property
    def example_input_array(self) -> BatchDict:
        raise NotImplementedError()

    @property
    def required_keys_train(self) -> DatasetConfigsRequiredKeys:
        raise NotImplementedError()

    @property
    def required_keys_eval(self) -> DatasetConfigsRequiredKeys:
        raise NotImplementedError()

    def forward_impl(self, batch: BatchDict) -> Union[BatchDict, LossDict]:
        raise NotImplementedError()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        raise NotImplementedError()

    def forward(self, batch: BatchDict) -> Union[BatchDict, LossDict]:
        context = contextlib.nullcontext() if self.training else torch.no_grad()
        with context:
            return self.forward_impl(batch)

    def training_step(self, batch: BatchDict) -> LossDict:
        loss: LossDict = self.forward(batch)
        self.log_dict(dict(**loss), on_step=True, prog_bar=True)
        return self.forward(batch)

    def validation_step(self, batch: BatchDict) -> None:
        pred: BatchDict = self.forward(batch)
        if not self.trainer.sanity_checking:
            self.metric.update(batch, pred)
        return None

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            metric = self.metric.compute()
            self.log_dict(dict(**metric), on_epoch=True, prog_bar=True)
            self.metric.reset()
        return None

    def test_step(self, batch: BatchDict) -> None:
        return self.validation_step(batch)

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()
