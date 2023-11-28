import json
import platform
import time
from pathlib import Path
from typing import List, Optional, Union, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.base_classes import BaseSystem
from ravt.core.functional_classes import LocalDataModule
from ravt.core import configs

from .callbacks import VisualizeCallback, NewTqdmProgressBar
from .train import load_ckpt


def run_test(
        system: BaseSystem,
        exp_tag: str,
        batch_size: int,
        num_workers: int = 0,
        device_ids: Optional[List[int]] = None,
        debug: bool = __debug__,
        seed: Optional[int] = None,
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
        NewTqdmProgressBar(),
    ]
    if callback_visualize:
        callbacks.append(VisualizeCallback(
            exp_tag, (600, 960), display_mode='save_video', batch_mode='all',
            visualize_eval_interval=1, visualize_test_interval=1,
        ))

    # Load
    load_ckpt(system, ckpt_dir, resume)

    # Log parameters
    variables = locals()
    exp_settings = {
        name: variables[name]
        for name in ['exp_tag', 'batch_size', 'num_workers', 'device_ids', 'debug', 'seed', 'resume']
    }
    logger.info(f'system hparams:\n{json.dumps(system.hparams, indent=2)}')
    logger.info(f'exp settings:\n{json.dumps(exp_settings, indent=2)}')
    train_log_dir.joinpath(f'test_{exp_tag}_{time_tag}.log').write_text(
        f'{json.dumps(system.hparams, indent=2)}\n{json.dumps(exp_settings, indent=2)}'
    )

    trainer = pl.Trainer(
        default_root_dir=train_log_dir,
        accelerator='gpu',
        devices=device_ids,
        callbacks=callbacks,
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if platform.system() == 'Linux' and len(device_ids) > 1
            else 'auto'
        ),
        precision='16-mixed',
        enable_checkpointing=False,
        limit_test_batches=10 if debug else None,
        logger=False,
    )

    trainer.test(system, datamodule=dataset)

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        # keyboard exit
        logger.warning('Ctrl+C detected. Shutting down training procedure.')
        raise KeyboardInterrupt()

    return {
        'test_mAP': round(trainer.logged_metrics['mAP'].item(), 5),
    }
