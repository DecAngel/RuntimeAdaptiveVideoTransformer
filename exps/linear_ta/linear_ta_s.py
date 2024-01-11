import json
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from typing import Optional, Literal, List, Union

import torch
import fire
import pytorch_lightning as pl

from ravt.data_sources import ArgoverseDataSource
from ravt.systems.yolox import linear_ta_s
from ravt.core.launchers.train import run_train
from ravt.core.launchers.test import run_test

torch.set_float32_matmul_precision('high')


def main(
        exp_tag: str, past_time_constant: List[int], future_time_constant: List[int],
        train: bool = True, batch_size: Optional[int] = None, device_id: int = 0,
        enable_cache: bool = True, seed: Optional[int] = None, debug: bool = False,
        visualize_mode: Optional[Literal['show_opencv', 'show_plt', 'write_image', 'write_video']] = None,
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        train_scheduler: Literal['yolox', 'msca'] = 'yolox',
):
    """ Train or test linear_ta_s model on Argoverse-HD

    :param train_scheduler:
    :param backbone:
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
    system = linear_ta_s(
        data_sources={
            'train': ArgoverseDataSource('train', enable_cache=enable_cache),
            'eval': ArgoverseDataSource('eval', enable_cache=enable_cache),
            'test': ArgoverseDataSource('test', enable_cache=enable_cache),
        },
        batch_size=batch_size,
        num_workers=num_workers,
        past_time_constant=past_time_constant,
        future_time_constant=future_time_constant,
        backbone=backbone,
        train_scheduler=train_scheduler,
        num_classes=8,
        lr=0.001 / 64 * (batch_size or 2),
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    if train:
        if backbone == 'pafpn':
            system.load_from_pth(
                Path(root_dir) / 'weights' / 'pretrained' / 'yolox_s.pth'
            )
        else:
            system.load_from_pth(
                Path(root_dir) / 'weights' / 'pretrained' / 'yolox_s_drfpn.pth'
            )
        res = run_train(
            system, exp_tag=exp_tag, max_epoch=15, device_ids=[device_id], resume=None,
            debug=debug, visualize_mode=visualize_mode,
        )
    else:
        system.load_from_ckpt(
            '/home/xzhang1048576/projects/ravt/outputs/checkpoints/linear_ta_s_-3-2-101_pafpn_20_win10/mAP=0.25687_091935.ckpt'
        )
        res = run_test(
            system, exp_tag=exp_tag, device_ids=[device_id], resume=None,
            debug=debug, visualize_mode=visualize_mode,
        )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    fire.Fire(main)
