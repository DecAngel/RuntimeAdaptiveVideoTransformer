import contextlib
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.optim import Optimizer

from .data_source import BaseDataSource
from .data_sampler import BaseDataSampler
from .transform import BaseTransform
from .metric import BaseMetric
from .sap_strategy import BaseSAPStrategy


try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # low pytorch version where LRScheduler is protected
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..constants import (
    BatchDict, PredDict, LossDict, ImageInferenceType, BBoxesInferenceType,
    OutputDict,
)
from ..utils.lightning_logger import ravt_logger as logger


class BaseSystem(pl.LightningModule):
    def __init__(
            self,
            data_source: Optional[BaseDataSource] = None,
            data_sampler: Optional[BaseDataSampler] = None,
            preprocess: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
            strategy: Optional[BaseSAPStrategy] = None,
    ):
        super().__init__()
        self.data_source = data_source
        self.data_sampler = data_sampler
        self.preprocess = preprocess
        self.metric = metric
        self.strategy = strategy

    def load_from_pth(self, file_path: Union[str, Path]) -> None:
        state_dict = torch.load(str(file_path), map_location='cpu')
        with contextlib.suppress(NotImplementedError):
            state_dict = self.pth_adapter(state_dict)

        misshaped_keys = []
        ssd = self.state_dict()
        for k in list(state_dict.keys()):
            if k in ssd and ssd[k].shape != state_dict[k].shape:
                misshaped_keys.append(k)
                del state_dict[k]

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = list(filter(lambda key: key not in misshaped_keys, missing_keys))

        if len(missing_keys) > 0:
            logger.warning(f'Missing keys in ckpt: {missing_keys}')
        if len(unexpected_keys) > 0:
            logger.warning(f'Unexpected keys in ckpt: {unexpected_keys}')
        if len(misshaped_keys) > 0:
            logger.warning(f'Misshaped keys in ckpt: {misshaped_keys}')
        logger.info(f'pth file {file_path} loaded!')

    def pth_adapter(self, state_dict: Dict) -> Dict:
        raise NotImplementedError()

    def load_from_ckpt(self, file_path: Union[str, Path], strict: bool = True):
        self.load_state_dict(
            torch.load(
                str(file_path),
                map_location=self.device
            )['state_dict'],
            strict=strict,
        )
        logger.info(f'ckpt file {file_path} loaded!')

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        raise NotImplementedError()

    def inference_impl(
            self,
            image: ImageInferenceType,
            buffer: Optional[Dict],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BBoxesInferenceType, Optional[Dict]]:
        raise NotImplementedError()

    def inference(
            self,
            image: ImageInferenceType,
            buffer: Optional[Dict],
            past_time_constant: Optional[List[int]] = None,
            future_time_constant: Optional[List[int]] = None,
    ) -> Tuple[BBoxesInferenceType, Optional[Dict]]:
        with torch.inference_mode():
            return self.inference_impl(image, buffer, past_time_constant, future_time_constant)

    def forward_impl(
            self,
            batch: BatchDict,
    ) -> OutputDict:
        raise NotImplementedError()

    def forward(
            self,
            batch: BatchDict,
    ) -> OutputDict:
        with torch.inference_mode():
            batch = self.preprocess.transform(batch) if self.preprocess is not None else batch
        with (contextlib.nullcontext() if self.training else torch.inference_mode()):
            return self.forward_impl(batch)

    def training_step(self, batch: BatchDict, *args, **kwargs) -> OutputDict:
        output: OutputDict = self.forward(batch)
        self.log('loss', output['loss'], on_step=True, prog_bar=True)
        return output

    def on_validation_epoch_start(self) -> None:
        if self.metric is not None:
            self.metric.reset()

    def validation_step(self, batch: BatchDict, *args, **kwargs) -> OutputDict:
        output: OutputDict = self.forward(batch)
        if not self.trainer.sanity_checking and self.metric is not None:
            self.metric.update(batch, output)
        return output

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.metric is not None:
            metric = self.metric.compute()
            self.log_dict(dict(**metric), on_epoch=True, prog_bar=True)
            self.metric.reset()
        return None

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def test_step(self, batch: BatchDict, *args, **kwargs) -> OutputDict:
        return self.validation_step(batch)

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:

