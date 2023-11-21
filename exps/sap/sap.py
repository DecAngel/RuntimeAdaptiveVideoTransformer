import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from ravt.systems.sap_strategies import NormalStrategy, DynamicSchedulingStrategy, DadeSchedulingStrategy, MSCASchedulingStrategy
from ravt.systems.data_sources import ArgoverseDataSource
from ravt.systems.yolox import streamyolo_s, longshortnet_s, damo_streamnet_s, yolox_s, msca_s
from ravt.launchers.sap import run_sap

torch.set_float32_matmul_precision('highest')


def main(seed: Optional[int] = None):
    seed = pl.seed_everything(seed)
    device_id = 0
    sap_factor = 1.0
    exp_tag = 'MSCA_S with Dynamic scheduler on 1x sAP'

    system = msca_s(
        data_source=ArgoverseDataSource(),
        strategy=NormalStrategy(exp_tag),
        predict_num=1,
        num_classes=8,
        lr=0.001 / 64,
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    """
    system = msca_s(
        data_source=ArgoverseDataSource(),
        strategy=NormalStrategy(exp_tag=exp_tag),
        past_time_constant=[-1],
        future_time_constant=[1],
        backbone='drfpn',
        neck_type='ta',
        neck_act_type='relu',
        neck_p_init=0.0,
        neck_tpe_merge='add',
        neck_dropout=0.0,
        train_mask=False,
        train_scheduler='yolox',
        num_classes=8,
        lr=0.001 / 64,
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    """

    system.load_from_ckpt(
        Path(root_dir) / 'weights' / 'trained' / 'msca_s_012_mAP=0.31546_3342393340_131346.ckpt'
    )
    res = run_sap(
        system, sap_factor=sap_factor, dataset_resize_ratio=2, dataset_fps=30,
        device_id=device_id, seed=seed
    )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
