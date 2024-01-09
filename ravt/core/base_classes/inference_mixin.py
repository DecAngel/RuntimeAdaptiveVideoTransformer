import contextlib
from typing import Optional, Dict, List, Tuple, Protocol, Union

import torch
import pytorch_lightning as pl
from torch import nn

from ..constants import BatchNDict, BatchTDict, LossDict
from .transform import BaseTransform
from .metric import BaseMetric
from .sap_strategy import BaseSAPStrategy


class InferenceMixin:
    # training: bool
    # transform: BaseTransform
    # metric: BaseMetric
    # strategy: BaseSAPStrategy
    # trainer: pl.Trainer
    # def log(self, *args, **kwargs): ...
    # def log_dict(self, *args, **kwargs): ...

    @property
    def fraction_epoch(self):
        return self.trainer.global_step / self.trainer.estimated_stepping_batches

    def inference_impl(
            self,
            batch: BatchTDict,
            buffer: Optional[Dict],
    ) -> Tuple[BatchTDict, Optional[Dict]]:
        raise NotImplementedError()

    def inference(
            self,
            batch: BatchTDict,
            buffer: Optional[Dict],
    ) -> Tuple[BatchTDict, Optional[Dict]]:
        with torch.inference_mode():
            return self.inference_impl(batch, buffer)

    def forward_impl(
            self,
            batch: BatchTDict,
    ) -> Union[BatchTDict, LossDict]:
        raise NotImplementedError()

    def forward(
            self,
            batch: BatchTDict,
    ) -> Union[BatchTDict, LossDict]:
        with torch.inference_mode():
            batch = self.transform.preprocess_tensor(batch) if self.transform is not None else batch
        with contextlib.nullcontext() if self.training else torch.inference_mode():
            return self.forward_impl(batch)

    def training_step(self, batch: BatchTDict, *args, **kwargs) -> LossDict:
        output: LossDict = self.forward(batch)
        self.log_dict(output, on_step=True, prog_bar=True)
        return output

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.metric is not None:
            self.metric.reset()

    def validation_step(self, batch: BatchTDict, *args, **kwargs) -> BatchTDict:
        output: BatchTDict = self.forward(batch)
        if not self.trainer.sanity_checking and self.metric is not None:
            self.metric.update(batch, output)
        return output

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if not self.trainer.sanity_checking and self.metric is not None:
            metric = self.metric.compute()
            self.log_dict(dict(**metric), on_epoch=True, prog_bar=True)
            self.metric.reset()
        return None

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        return self.on_validation_epoch_start()

    def test_step(self, batch: BatchTDict, *args, **kwargs) -> BatchTDict:
        return self.validation_step(batch)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        return self.on_validation_epoch_end()

    @staticmethod
    def freeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(module: nn.Module):
        for param in module.parameters():
            param.requires_grad = True
