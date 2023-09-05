import functools
import platform
import datetime
from typing import List, TypedDict, Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ravt.utils.lightning_logger import ravt_logger as logger
from ravt.utils.hashtags import module_hash
from ravt.protocols.classes import BaseLauncher, BaseModel, BaseDataset
from ravt.protocols.structures import InternalConfigs, ConfigTypes


class TrainTestLauncher(BaseLauncher):
    def __init__(
            self,
            model: BaseModel,
            dataset: BaseDataset,
            configs: Optional[InternalConfigs] = None,
            device_ids: Optional[List[int]] = None,
            debug: bool = __debug__,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.configs = configs
        self.device_ids = device_ids or [0]
        self.debug = debug
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

    def train(self) -> TypedDict('TrainResult', {'eval_mAP': float}):
        self.phase_init(self.configs)
        trainer = pl.Trainer(
            default_root_dir=str(self.output_train_log_dir),
            accelerator='gpu',
            devices=self.device_ids,
            callbacks=[
                pl.callbacks.ModelSummary(max_depth=2),
                pl.callbacks.ModelCheckpoint(
                    monitor='mAP', mode='max', dirpath=str(self.output_ckpt_dir), filename=self.tag
                ),
                pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ],
            strategy=(
                DDPStrategy(find_unused_parameters=False)
                if platform.system() == 'Linux' and len(self.device_ids) > 1
                else 'auto'
            ),
            precision='16-mixed',
            max_epochs=2 if self.debug else 100,
            limit_train_batches=20 if self.debug else None,
            limit_val_batches=20 if self.debug else None,
            detect_anomaly=False if self.debug else False,
            profiler='simple' if self.debug else None,
        )
        trainer.fit(self.model, datamodule=self.dataset)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            logger.warning('Ctrl+C detected. Shutting down training procedure.')
            raise KeyboardInterrupt()

        return {
            'eval_mAP': round(trainer.logged_metrics['mAP'].item(), 3),
        }

    def test(self, from_best_ckpt: bool = True) -> TypedDict('TestResult', {'test_mAP': float}):
        self.phase_init(self.configs)
        trainer = pl.Trainer(
            default_root_dir=str(self.output_train_log_dir),
            accelerator='gpu',
            devices=self.device_ids,
            callbacks=[
                pl.callbacks.ModelSummary(max_depth=0),
            ],
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

        if from_best_ckpt:
            hash_tag = module_hash(self.model)
            for file in sorted(self.output_ckpt_dir.iterdir(), reverse=True):
                if hash_tag in file.name:
                    ckpt_path = str(file)
                    break
            else:
                raise FileNotFoundError(f'Cannot find checkpoint with hash tag {hash_tag}')
        else:
            ckpt_path = None

        trainer.test(self.model, datamodule=self.dataset, ckpt_path=ckpt_path)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            logger.warning('Ctrl+C detected. Shutting down testing procedure.')
            raise KeyboardInterrupt()

        return {'test_mAP': round(trainer.logged_metrics['mAP'].item(), 3)}
