import contextlib
from pathlib import Path
from typing import Union, Tuple, List, Optional, Callable, Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..constants import BatchKeys, BatchDict, PredDict, LossDict, MetricDict, PhaseTypes, AllConfigs
from ..utils.phase_init import PhaseInitMixin
from ..utils.lightning_logger import ravt_logger as logger


class BaseTransform(PhaseInitMixin, nn.Module):
    def forward(self, batch: BatchDict) -> BatchDict:
        raise NotImplementedError()


class BaseMetric(PhaseInitMixin, Metric):
    def update(self, batch: BatchDict, pred: PredDict) -> None:
        raise NotImplementedError()

    def compute(self) -> MetricDict:
        raise NotImplementedError()


class BaseSystem(PhaseInitMixin, pl.LightningModule):
    def __init__(
            self,
            pretrain_adapter: Optional[Callable[[Path], Dict]] = None,
            preprocess: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
    ):
        super().__init__()
        self.pretrain_adapter = pretrain_adapter
        self.preprocess = preprocess
        self.metric = metric

    @property
    def example_input_array(self) -> Tuple[BatchDict]:  # Avoid being dispatched to kwargs by lightning
        raise NotImplementedError()

    @property
    def required_keys_train(self) -> BatchKeys:
        raise NotImplementedError()

    @property
    def required_keys_eval(self) -> BatchKeys:
        raise NotImplementedError()

    @property
    def produced_keys(self) -> BatchKeys:
        raise NotImplementedError()

    def forward_impl(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        raise NotImplementedError()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        raise NotImplementedError()

    def inference_impl(self, batch: BatchDict) -> PredDict:
        return self.forward_impl(batch)

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'environment':
            configs['internal']['required_keys_train'] = self.required_keys_train
            configs['internal']['required_keys_eval'] = self.required_keys_eval
            configs['internal']['produced_keys'] = self.produced_keys
        if phase == 'model' and self.pretrain_adapter is not None:
            pretrained_dir = configs['environment']['weight_pretrained_dir']
            state_dict = self.pretrain_adapter(pretrained_dir)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                logger.warning(f'Missing keys in ckpt: {missing_keys}')
            if len(unexpected_keys) > 0:
                logger.warning(f'Unexpected keys in ckpt: {unexpected_keys}')
        return configs

    def forward(self, batch: BatchDict) -> Union[PredDict, LossDict]:
        with torch.inference_mode():
            batch = self.preprocess(batch) if self.preprocess is not None else batch
        context = contextlib.nullcontext() if self.training else torch.inference_mode()
        with context:
            return self.forward_impl(batch)

    def inference(self, batch: BatchDict) -> PredDict:
        with torch.inference_mode():
            batch = self.preprocess(batch) if self.preprocess is not None else batch
            return self.inference_impl(batch)

    def training_step(self, batch: BatchDict, *args, **kwargs) -> LossDict:
        loss: LossDict = self.forward(batch)
        self.log_dict(dict(**loss), on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: BatchDict, *args, **kwargs) -> PredDict:
        pred: PredDict = self.forward(batch)
        if not self.trainer.sanity_checking and self.metric is not None:
            self.metric.update(batch, pred)
        return pred

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.metric is not None:
            metric = self.metric.compute()
            self.log_dict(dict(**metric), on_epoch=True, prog_bar=True)
            self.metric.reset()
        return None

    def test_step(self, batch: BatchDict, *args, **kwargs) -> PredDict:
        return self.validation_step(batch)

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()