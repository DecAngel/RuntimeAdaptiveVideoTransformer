import json
import platform
import time
from pathlib import Path
from typing import List, Optional, Union, Literal

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ..utils.image_writer import ImageWriter
from ..utils.lightning_logger import ravt_logger as logger
from ..utils.progress_bar import NewTqdmProgressBar
from ..base_classes import BaseSystem
from .. import configs
from .train import load_ckpt


def run_test(
        system: BaseSystem,
        exp_tag: str,
        device_ids: Optional[List[int]] = None,
        debug: bool = __debug__,
        resume: Union[Path, str, Literal['last', 'best'], None] = None,
        visualize_mode: Optional[Literal['show_opencv', 'show_plt', 'write_image', 'write_video']] = None,
):
    # Init
    device_ids = device_ids or [0]
    time_tag = time.strftime("%d%H%M")
    system.eval()

    # Paths
    s = set(locals().keys())
    train_log_dir = configs.output_train_log_dir.joinpath(exp_tag)
    ckpt_dir = configs.output_ckpt_dir.joinpath(exp_tag)
    visualize_dir = configs.output_visualize_dir.joinpath(exp_tag)
    s = set(locals().keys()) - s - set('s')
    for name in s:
        locals()[name].mkdir(exist_ok=True, parents=True)

    # Visualization Writer
    if visualize_mode is not None:
        system.image_writer = ImageWriter(tag=exp_tag, mode=visualize_mode, visualization_dir=visualize_dir)

    # Callbacks
    callbacks = [
        pl.callbacks.ModelSummary(max_depth=2),
        NewTqdmProgressBar(),
    ]

    # Load
    load_ckpt(system, ckpt_dir, resume)

    # Log parameters
    variables = locals()
    exp_settings = {
        name: variables[name]
        for name in ['exp_tag', 'device_ids', 'debug', 'resume', 'visualize_mode']
    }
    all_settings = json.dumps({'system_hparams': system.hparams, 'exp_settings': exp_settings}, indent=2)
    logger.info(all_settings)
    train_log_dir.joinpath(f'train_{exp_tag}_{time_tag}.log').write_text(all_settings)

    trainer = pl.Trainer(
        default_root_dir=train_log_dir,
        logger=False,
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
        log_every_n_steps=1 if debug else 100,
    )

    trainer.test(system)

    if visualize_mode is not None:
        system.image_writer.close()

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        # keyboard exit
        logger.warning('Ctrl+C detected. Shutting down training procedure.')
        raise KeyboardInterrupt()

    return {
        'test_mAP': round(trainer.logged_metrics['mAP'].item(), 5),
    }
