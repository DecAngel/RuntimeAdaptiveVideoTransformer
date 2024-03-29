import json
import os
import sys
from pathlib import Path

from ravt.core.launchers.test import run_test

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from typing import Optional, Literal

import torch
import fire
import pytorch_lightning as pl

from ravt.data_sources import ArgoverseDataSource
from ravt.systems.yolox import yolox_l
from ravt.core.launchers.train import run_train

torch.set_float32_matmul_precision('high')


def main(
        exp_tag: str, predict_num: int = 0, train: bool = True, batch_size: Optional[int] = None, device_id: int = 0,
        enable_cache: bool = True, seed: Optional[int] = None, debug: bool = False,
        visualize_mode: Optional[Literal['show_opencv', 'write_image', 'write_video']] = None,
):
    """ Train or test yolox_l model on Argoverse-HD.

    :param exp_tag: the tag for the experiment
    :param predict_num: predict offset for the model
    :param train: train or test
    :param batch_size: batch size of the exp
    :param device_id: the gpu id
    :param enable_cache: use shared memory allocator
    :param seed: the random seed
    :param debug: enable debug mode
    :param visualize_mode: choose visualization mode
    :return:
    """
    seed = pl.seed_everything(seed)
    batch_size = 4 if debug else batch_size
    num_workers = 0 if debug else 8
    system = yolox_l(
        data_sources={
            'train': ArgoverseDataSource('train', enable_cache=enable_cache),
            'eval': ArgoverseDataSource('eval', enable_cache=enable_cache),
            'test': ArgoverseDataSource('test', enable_cache=enable_cache),
        },
        batch_size=batch_size,
        num_workers=num_workers,
        predict_num=predict_num,
        num_classes=8,
        lr=0.001 / 64 * (batch_size or 2),
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    if train:
        system.load_from_pth(
            Path(root_dir) / 'weights' / 'pretrained' / 'yolox_l.pth'
        )
        res = run_train(
            system, exp_tag=exp_tag, max_epoch=15, device_ids=[device_id], resume=None,
            debug=debug, visualize_mode=visualize_mode,
        )
    else:
        system.load_from_ckpt(
            Path(root_dir) / 'weights' / 'trained' / 'yolox_s_00_mAP=0.31668_3342393340_050132.ckpt'
        )
        res = run_test(
            system, exp_tag=exp_tag, device_ids=[device_id], resume=None,
            debug=debug, visualize_mode=visualize_mode,
        )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    fire.Fire(main)
