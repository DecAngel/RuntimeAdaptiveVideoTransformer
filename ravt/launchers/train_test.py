import functools
import platform
import time
from pathlib import Path
from typing import List, TypedDict, Optional, Union, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.tuner import Tuner

from ravt.core.constants import AllConfigs, PhaseTypes
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.hashtags import hash_all
from ravt.core.base_classes import BaseLauncher, BaseSystem, BaseDataSource, launcher_entry
from ravt.core.functional_classes import LocalDataModule

from ..base_configs import environment_configs
from .callbacks import EMACallback, VisualizeCallback, NewTqdmProgressBar


class TrainTestLauncher(BaseLauncher):
    def __init__(
            self,
            system: BaseSystem,
            data_source: BaseDataSource,
            exp_tag: str,
            envs: Optional[AllConfigs] = None,
            batch_size: Optional[int] = None,
            num_workers: int = 0,
            max_epoch: int = 10,
            device_ids: Optional[List[int]] = None,
            debug: bool = __debug__,
            seed: Optional[int] = None,
            callback_ema: bool = False,
            callback_visualize: bool = False,
    ):
        super().__init__(envs or environment_configs)

        self.batch_size = batch_size
        self.model = system
        self.dataset = LocalDataModule(data_source, batch_size or 2, num_workers)
        self.exp_tag = exp_tag
        self.max_epoch = max_epoch
        self.device_ids = device_ids or [0]
        self.debug = debug
        self.seed = pl.seed_everything(seed)
        self.dir_changed: bool = False

        self.callback_ema = EMACallback() if callback_ema else None
        self.callback_visualize = VisualizeCallback(
            (300, 480), mode='save', visualize_eval_interval=100, visualize_test_interval=100,
        ) if callback_visualize else None
        self.train_callbacks = filter(None, [self.callback_ema, self.callback_visualize])
        self.test_callbacks = filter(None, [self.callback_visualize])

        self.output_ckpt_dir = None
        self.output_train_log_dir = None

    @functools.cached_property
    def parameter_tag(self):
        return hash_all(
            self.model, self.dataset, self.callback_ema
        )

    @functools.cached_property
    def time_tag(self):
        return time.strftime('%d%H%M')

    def log_exp_info(self):
        logger.info(
            f'exp_tag: {self.exp_tag}'
            f'\ttime_tag: {self.time_tag}'
            f'\tparameter_tag: {self.parameter_tag}'
            f'\tseed: {self.seed}'
        )

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'environment':
            configs['internal']['exp_tag'] = self.exp_tag
            if not self.dir_changed:
                configs['environment']['output_ckpt_dir'] /= self.exp_tag
                configs['environment']['output_train_log_dir'] /= self.exp_tag
                configs['environment']['output_visualize_dir'] /= self.exp_tag
                configs['environment']['output_result_dir'] /= self.exp_tag
                configs['environment']['output_sap_log_dir'] /= self.exp_tag
                configs['environment']['output_ckpt_dir'].mkdir(exist_ok=True)
                configs['environment']['output_train_log_dir'].mkdir(exist_ok=True)
                configs['environment']['output_visualize_dir'].mkdir(exist_ok=True)
                configs['environment']['output_result_dir'].mkdir(exist_ok=True)
                configs['environment']['output_sap_log_dir'].mkdir(exist_ok=True)
                self.dir_changed = True
        elif phase == 'launcher':
            self.output_train_log_dir = configs['environment']['output_train_log_dir']
            self.output_ckpt_dir = configs['environment']['output_ckpt_dir']
        return configs

    def load_ckpt(self, ckpt: Union[Path, str, Literal['last', 'best'], None]):
        ckpt_path = None
        if ckpt in ['last', 'best']:
            for file in sorted(self.output_ckpt_dir.iterdir(), reverse=True):
                if (ckpt == 'best' and self.parameter_tag in file.name) or (ckpt == 'last' and 'last' in file.name):
                    ckpt_path = str(file)
                    break
            else:
                logger.warning(f'{ckpt} ckpt not found in {self.output_ckpt_dir}')
        elif isinstance(ckpt, (Path, str)):
            ckpt_path = str(ckpt)

        if ckpt_path is not None:
            d = torch.load(ckpt_path)
            if self.callback_ema is not None:
                # load ema state dict
                logger.info(f'load ema state dict from {ckpt_path}')
                self.model.load_state_dict(d['callbacks']['EMACallback']['state_dict_ema'])
            else:
                logger.info(f'load state dict from {ckpt_path}')
                self.model.load_state_dict(d['state_dict'])

    @launcher_entry({'internal': {'stage': 'fit'}})
    def train(self, resume: Union[Path, str, Literal['last', 'best'], None]) -> TypedDict('TrainResult', {'eval_mAP': float}):
        self.log_exp_info()
        callbacks = [
            pl.callbacks.ModelSummary(max_depth=5),
            pl.callbacks.ModelCheckpoint(
                save_last=True, monitor='mAP', mode='max', dirpath=str(self.output_ckpt_dir),
                filename=f'{self.time_tag}_{{mAP:.5f}}_{self.parameter_tag}_{self.seed}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            NewTqdmProgressBar(),
        ]
        callbacks.extend(self.train_callbacks)
        self.load_ckpt(resume)

        trainer = pl.Trainer(
            default_root_dir=self.output_train_log_dir,
            logger=TensorBoardLogger(save_dir=self.output_train_log_dir, version=f'{self.time_tag}_{self.parameter_tag}'),
            accelerator='gpu',
            devices=self.device_ids,
            callbacks=callbacks,
            strategy=(
                DDPStrategy(find_unused_parameters=False)
                if platform.system() == 'Linux' and len(self.device_ids) > 1
                else 'auto'
            ),
            precision='16-mixed',
            max_epochs=2 if self.debug else self.max_epoch,
            limit_train_batches=10 if self.debug else None,
            limit_val_batches=10 if self.debug else None,
            detect_anomaly=True if self.debug else False,
            profiler='simple' if self.debug else None,
        )
        if self.batch_size is None:
            # tuning
            tuner = Tuner(trainer)
            original_batch_size = self.dataset.hparams.batch_size
            final_batch_size = tuner.scale_batch_size(self.model, datamodule=self.dataset, mode='power', method='fit')
            final_batch_size = max(final_batch_size - 2, 1)
            accumulate_gradient = max(original_batch_size // final_batch_size, 1)
            self.dataset.hparams.batch_size = final_batch_size
            trainer.accumulate_grad_batches = accumulate_gradient
            logger.info(f'Tuner setting batch_size to {final_batch_size}, '
                        f'accumulate gradient per {accumulate_gradient} batch(es)')

        trainer.fit(self.model, datamodule=self.dataset)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            logger.warning('Ctrl+C detected. Shutting down training procedure.')
            raise KeyboardInterrupt()

        return {
            'eval_mAP': round(trainer.checkpoint_callback.best_model_score.item(), 3),
        }

    @launcher_entry({'internal': {'stage': 'test'}})
    def test(self, resume: Union[Path, str, Literal['last', 'best'], None]) -> TypedDict('TestResult', {'test_mAP': float}):
        self.log_exp_info()
        callbacks = [
            pl.callbacks.ModelSummary(max_depth=2),
            NewTqdmProgressBar(),
        ]
        callbacks.extend(self.test_callbacks)
        self.load_ckpt(resume)

        trainer = pl.Trainer(
            default_root_dir=self.output_train_log_dir,
            accelerator='gpu',
            devices=self.device_ids,
            callbacks=callbacks,
            strategy=(
                DDPStrategy(find_unused_parameters=False)
                if platform.system() == 'Linux' and len(self.device_ids) > 1
                else 'auto'
            ),
            precision='16-mixed',
            enable_checkpointing=False,
            limit_test_batches=10 if self.debug else None,

            logger=False,
        )
        if self.batch_size is None:
            # tuning
            tuner = Tuner(trainer)
            final_batch_size = tuner.scale_batch_size(self.model, datamodule=self.dataset, mode='power', method='test')
            final_batch_size = max(final_batch_size - 2, 1)
            self.dataset.hparams.batch_size = final_batch_size
            logger.info(f'Tuner setting batch_size to {final_batch_size}')

        trainer.test(self.model, datamodule=self.dataset)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            logger.warning('Ctrl+C detected. Shutting down testing procedure.')
            raise KeyboardInterrupt()

        return {'test_mAP': round(trainer.logged_metrics['mAP'].item(), 3)}
