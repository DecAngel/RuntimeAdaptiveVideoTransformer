import functools
import platform
import datetime
from typing import List, TypedDict, Optional, Union, Literal

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ravt.utils.lightning_logger import ravt_logger as logger
from ravt.utils.hashtags import module_hash
from ravt.protocols.classes import BaseLauncher, BaseModel, BaseDataset
from ravt.protocols.structures import InternalConfigs, ConfigTypes, EnvironmentConfigs

from .callbacks import EMACallback, VisualizeCallback


class TrainTestLauncher(BaseLauncher):
    def __init__(
            self,
            model: BaseModel,
            dataset: BaseDataset,
            envs: Optional[EnvironmentConfigs] = None,
            device_ids: Optional[List[int]] = None,
            debug: bool = __debug__,
            callback_ema: bool = False,
            callback_visualize: bool = False,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.envs = envs or {}
        self.device_ids = device_ids or [0]
        self.debug = debug

        self.callback_ema = EMACallback() if callback_ema else None
        self.callback_visualize = VisualizeCallback(
            (300, 480), mode='save', visualize_eval_interval=100, visualize_test_interval=100,
        ) if callback_visualize else None
        self.additional_callbacks = [c for c in [self.callback_ema, self.callback_visualize] if c is not None]

        self.output_ckpt_dir = None
        self.output_train_log_dir = None

    @functools.cached_property
    def tag(self):
        return '_'.join([
            datetime.datetime.now().strftime('%y%m%d_%H%M%S'),
            self.dataset.__class__.__name__[:6],
            self.model.__class__.__name__[:6],
            module_hash(self.dataset),
            module_hash(self.model),
        ])

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'launcher':
            self.output_train_log_dir = configs['environment']['output_train_log_dir']
            self.output_ckpt_dir = configs['environment']['output_ckpt_dir']
        return configs

    def get_last_ckpt(self) -> Optional[str]:
        hash_tag = module_hash(self.model)
        for file in sorted(self.output_ckpt_dir.iterdir(), reverse=True):
            if hash_tag in file.name:
                return str(file)
        else:
            return None

    def train(self, resume: Union[str, Literal['last'], None] = None) -> TypedDict('TrainResult', {'eval_mAP': float}):
        self.phase_init(self.envs)
        callbacks = [
            pl.callbacks.ModelSummary(max_depth=2),
            pl.callbacks.ModelCheckpoint(
                monitor='mAP', mode='max', dirpath=str(self.output_ckpt_dir), filename=self.tag
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ]
        callbacks.extend(self.additional_callbacks)
        ckpt_path = self.get_last_ckpt() if resume == 'last' else resume

        trainer = pl.Trainer(
            default_root_dir=str(self.output_train_log_dir),
            accelerator='gpu',
            devices=self.device_ids,
            callbacks=callbacks,
            strategy=(
                DDPStrategy(find_unused_parameters=False)
                if platform.system() == 'Linux' and len(self.device_ids) > 1
                else 'auto'
            ),
            precision='16-mixed',
            max_epochs=2 if self.debug else 15,
            limit_train_batches=20 if self.debug else None,
            limit_val_batches=20 if self.debug else None,
            detect_anomaly=False if self.debug else False,
            profiler='simple' if self.debug else None,
        )
        trainer.fit(self.model, datamodule=self.dataset, ckpt_path=ckpt_path)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            logger.warning('Ctrl+C detected. Shutting down training procedure.')
            raise KeyboardInterrupt()

        return {
            'eval_mAP': round(trainer.logged_metrics['mAP'].item(), 3),
        }

    def test(self, resume: Union[str, Literal['last'], None] = None) -> TypedDict('TestResult', {'test_mAP': float}):
        self.phase_init(self.envs)
        callbacks = [
            pl.callbacks.ModelSummary(max_depth=2),
        ]
        callbacks.extend(self.additional_callbacks)
        ckpt_path = self.get_last_ckpt() if resume == 'last' else resume

        trainer = pl.Trainer(
            default_root_dir=str(self.output_train_log_dir),
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
            limit_test_batches=20 if self.debug else None,
            logger=False,
        )

        trainer.test(self.model, datamodule=self.dataset, ckpt_path=ckpt_path)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            logger.warning('Ctrl+C detected. Shutting down testing procedure.')
            raise KeyboardInterrupt()

        return {'test_mAP': round(trainer.logged_metrics['mAP'].item(), 3)}
