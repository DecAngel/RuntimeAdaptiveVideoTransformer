import platform
import datetime
from typing import Dict, List, TypedDict, Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerStatus

from ravt.common.lightning_logger import ravt_logger as logger
from ravt.common.hashtags import module_hash
from ravt.configs import output_train_log_dir, output_ckpt_dir, weight_pretrained_dir


def run_train(
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        device_ids: Optional[List[int]] = None,
        debug: bool = __debug__,
        **kwargs,
) -> TypedDict('OutputTypedDict', {
    'val_mAP': float,
}):
    tag = '_'.join([
        datetime.datetime.now().strftime('%y%m%d_%H%M%S'),
        data_module.__class__.__name__[:6],
        model.__class__.__name__[:6],
        module_hash(data_module),
        module_hash(model),
    ])
    device_ids = device_ids or [0]
    trainer = pl.Trainer(
        default_root_dir=str(output_train_log_dir),
        accelerator='gpu',
        devices=device_ids,
        callbacks=[
            pl.callbacks.ModelSummary(max_depth=2),
            pl.callbacks.ModelCheckpoint(monitor='mAP', mode='max', dirpath=str(output_ckpt_dir), filename=tag),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if platform.system() == 'Linux' and len(device_ids) > 1
            else 'auto'
        ),
        precision=16,
        max_epochs=3 if debug else 100,
        limit_train_batches=10 if debug else None,
        limit_val_batches=10 if debug else None,
        detect_anomaly=False if debug else False,
        profiler='simple' if debug else None,
    )

    trainer.fit(model, datamodule=data_module)

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        # keyboard exit
        logger.warning('Ctrl+C detected. Shutting down training procedure.')
        raise KeyboardInterrupt()

    return {
        'val_mAP': round(trainer.logged_metrics['mAP'].item(), 3),
    }


def run_test(
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        from_best_ckpt: bool = True,
        device_ids: Optional[List[int]] = None,
        debug: bool = __debug__,
        **kwargs,
) -> TypedDict('OutputTypedDict', {
    'val_mAP': float,
}):
    device_ids = device_ids or [0]
    trainer = pl.Trainer(
        default_root_dir=str(output_train_log_dir),
        accelerator='gpu',
        devices=device_ids,
        callbacks=[
            pl.callbacks.ModelSummary(max_depth=0),
        ],
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if platform.system() == 'Linux' and len(device_ids) > 1
            else 'auto'
        ),
        precision=16,
        enable_checkpointing=False,
        limit_test_batches=10 if debug else None,
        logger=False,
    )

    if from_best_ckpt:
        hash_tag = module_hash(model)
        for file in sorted(output_ckpt_dir.iterdir(), reverse=True):
            if hash_tag in file.name:
                ckpt_path = str(file)
                break
        else:
            raise FileNotFoundError(f'Cannot find checkpoint with hash tag {hash_tag}')
    else:
        ckpt_path = None

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        # keyboard exit
        logger.warning('Ctrl+C detected. Shutting down testing procedure.')
        raise KeyboardInterrupt()

    return round(trainer.logged_metrics['mAP'].item(), 3)


def run_train_and_test(
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        device_ids: Optional[List[int]] = None,
        debug: bool = __debug__,
):
    run_train(model=model, data_module=data_module, device_ids=device_ids, debug=debug)
    run_test(model=model, data_module=data_module, from_best_ckpt=True, device_ids=device_ids, debug=debug)
