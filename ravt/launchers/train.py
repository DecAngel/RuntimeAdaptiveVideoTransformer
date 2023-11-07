import json
import platform
import time
from pathlib import Path
from typing import List, Optional, Union, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.occupy_memory import occupy_mem
from ravt.core.base_classes import BaseSystem, BaseDataSource
from ravt.core.functional_classes import LocalDataModule
from ravt.core import configs

from .callbacks import EMACallback, VisualizeCallback, NewTqdmProgressBar


def load_ckpt(model: BaseSystem, ckpt_dir: Path, ckpt: Union[Path, str, Literal['last', 'best'], None]):
    ckpt_path = None
    if ckpt in ['last', 'best']:
        for file in sorted(ckpt_dir.iterdir(), reverse=True):
            if (ckpt == 'best') or (ckpt == 'last' and 'last' in file.name):
                ckpt_path = str(file)
                break
        else:
            logger.warning(f'{ckpt} ckpt not found in {ckpt_dir}')
    elif isinstance(ckpt, (Path, str)):
        ckpt_path = str(ckpt)

    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        logger.info(f'load state dict from {ckpt_path}')
        model.load_state_dict(d['state_dict'])


def run_train(
        system: BaseSystem,
        exp_tag: str,
        batch_size: int,
        num_workers: int = 0,
        max_epoch: int = 10,
        device_ids: Optional[List[int]] = None,
        debug: bool = __debug__,
        seed: Optional[int] = None,
        callback_ema: bool = False,
        callback_visualize: bool = False,
        resume: Union[Path, str, Literal['last', 'best'], None] = None,
):
    # Init
    if seed is None:
        raise ValueError('seed must be set at the beginning of the exp!')
    device_ids = device_ids or [0]
    dataset = LocalDataModule(system.data_source, system.data_sampler, batch_size, num_workers)
    time_tag = time.strftime("%d%H%M")

    # Paths
    s = set(locals().keys())
    train_log_dir = configs.output_train_log_dir.joinpath(exp_tag)
    ckpt_dir = configs.output_ckpt_dir.joinpath(exp_tag)
    visualize_dir = configs.output_visualize_dir.joinpath(exp_tag)
    s = set(locals().keys()) - s - set('s')
    for name in s:
        locals()[name].mkdir(exist_ok=True, parents=True)

    # Callbacks
    callbacks = [
        pl.callbacks.ModelSummary(max_depth=2),
        pl.callbacks.ModelCheckpoint(
            save_last=True, monitor='mAP', mode='max', dirpath=str(ckpt_dir),
            filename=f'{{mAP:.5f}}_{seed}_{time_tag}'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        NewTqdmProgressBar(),
    ]
    if callback_ema:
        callbacks.append(EMACallback())
    if callback_visualize:
        callbacks.append(VisualizeCallback(
            (300, 480), mode='save', visualize_eval_interval=100, visualize_test_interval=100,
        ))

    # Load
    load_ckpt(system, ckpt_dir, resume)

    # Log parameters
    variables = locals()
    exp_settings = {
        name: variables[name]
        for name in ['exp_tag', 'batch_size', 'num_workers', 'max_epoch', 'device_ids', 'debug', 'seed', 'resume']
    }
    logger.info(f'system hparams:\n{json.dumps(system.hparams, indent=2)}')
    logger.info(f'exp settings:\n{json.dumps(exp_settings, indent=2)}')

    trainer = pl.Trainer(
        default_root_dir=train_log_dir,
        logger=TensorBoardLogger(save_dir=train_log_dir, version=f'{time_tag}'),
        accelerator='gpu',
        devices=device_ids,
        callbacks=callbacks,
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if platform.system() == 'Linux' and len(device_ids) > 1
            else 'auto'
        ),
        precision='16-mixed',
        max_epochs=2 if debug else max_epoch,
        limit_train_batches=10 if debug else None,
        limit_val_batches=10 if debug else None,
        detect_anomaly=True if debug else False,
        profiler='simple' if debug else None,
    )

    trainer.fit(system, datamodule=dataset)

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        # keyboard exit
        logger.warning('Ctrl+C detected. Shutting down training procedure.')
        raise KeyboardInterrupt()

    return {
        'eval_mAP': round(trainer.checkpoint_callback.best_model_score.item(), 3),
    }
