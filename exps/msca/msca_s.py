import json
import os
import sys
from pathlib import Path

from ravt.launchers.test import run_test

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from typing import Literal, Optional, List

import torch
import fire
import pytorch_lightning as pl

from ravt.systems.data_sources import ArgoverseDataSource
from ravt.systems.yolox import msca_s
from ravt.launchers.train import run_train

torch.set_float32_matmul_precision('high')


def main(
        exp_tag: str, past_time_constant: List[int], future_time_constant: List[int],
        backbone: Literal['pafpn', 'drfpn'] = 'pafpn',
        neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'none',
        neck_p_init: Optional[float] = None,
        neck_dropout: float = 0.5,
        train_mask: bool = False,
        enable_cache: bool = True, seed: Optional[int] = None,
        batch_size: Optional[int] = None, device_id: int = 0, visualize: bool = False, debug: bool = False
):
    """ Train and test msca_s model on Argoverse-HD

    :param exp_tag: the tag for the experiment
    :param past_time_constant:
    :param future_time_constant:
    :param neck_act_type:
    :param neck_dropout:
    :param neck_p_init:
    :param backbone:
    :param train_mask:
    :param enable_cache: use shared memory allocator
    :param seed: the random seed
    :param batch_size: batch size of the exp, set None to auto-detect
    :param device_id: the cuda device id to place the model on
    :param visualize: enable visualization
    :param debug: enable debug mode
    :return:
    """
    seed = pl.seed_everything(seed)
    batch_size = 4 if debug else batch_size
    num_workers = 0 if debug else 8
    system = msca_s(
        data_source=ArgoverseDataSource(enable_cache=enable_cache),
        past_time_constant=past_time_constant,
        future_time_constant=future_time_constant,
        backbone=backbone,
        neck_act_type=neck_act_type,
        neck_p_init=neck_p_init,
        neck_dropout=neck_dropout,
        train_mask=train_mask,
        num_classes=8,
        lr=0.001 / 64 * (batch_size or 2),
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    train = True
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
            system, exp_tag=exp_tag, max_epoch=15,
            batch_size=batch_size, num_workers=num_workers, device_ids=[device_id], debug=debug,
            callback_ema=True, callback_visualize=visualize, resume=None, seed=seed,
        )
    else:
        """
        system.load_from_ckpt(
            Path(root_dir) / 'weights' / 'trained' / 'msca_s_-3-2-101_mAP=0.29661_232218_8a321e_470238574.ckpt'
        )
        """
        res = run_test(
            system, exp_tag=exp_tag,
            batch_size=batch_size, num_workers=num_workers, device_ids=[device_id], debug=debug,
            callback_visualize=visualize, resume='best', seed=seed,
        )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    fire.Fire(main)
