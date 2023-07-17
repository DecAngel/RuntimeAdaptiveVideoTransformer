import platform
import random
from typing import Optional, List

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ravt.common.lightning_logger import ravt_logger as logger
from ravt.configs import output_train_log_dir, output_visualize_dir


def run_visualize(
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        tag: Optional[str] = None,
        limit_batches: Optional[int] = 100,
        devices: Optional[List[int]] = None,
) -> None:
    if tag is None:
        filename = f'no_tag-{random.randint(0, 9999999999)}'
    else:
        filename = tag
    devices = devices or [0]
    visualize_dir = output_visualize_dir.joinpath(filename)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=str(output_train_log_dir),
        accelerator='gpu',
        devices=devices,
        callbacks=[
            pl.callbacks.ModelSummary(max_depth=0),
        ],
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if platform.system() == 'Linux' and len(devices) > 1
            else None
        ),
        precision=16,
        enable_checkpointing=False,
        limit_test_batches=limit_batches,
        logger=False,
    )

    batch_size = data_module.batch_size
    data_module.batch_size = 1
    model.visualize_dir = visualize_dir
    trainer.test(model, datamodule=data_module)
    model.visualize_dir = None
    data_module.batch_size = batch_size

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        # keyboard exit
        logger.warning('Ctrl+C detected. Shutting down testing procedure.')
        raise KeyboardInterrupt()

    return
