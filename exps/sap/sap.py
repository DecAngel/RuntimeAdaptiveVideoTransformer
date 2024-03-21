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

from ravt.sap_strategies import NormalStrategy, DynamicSchedulingStrategy
from ravt.data_sources import ArgoverseDataSource
from ravt.systems.yolox import msca_s, yolox_s
from ravt.core.launchers.sap import run_sap

torch.set_float32_matmul_precision('highest')


def main(seed: Optional[int] = None):
    seed = pl.seed_everything(seed)
    device_id = 0
    sap_factor = 1.0
    exp_tag = 'YOLOX_S with Dynamic scheduler on 1x sAP'

    system = yolox_s(
        data_sources={
            'train': ArgoverseDataSource('train', enable_cache=False),
            'eval': ArgoverseDataSource('eval', enable_cache=False),
            'test': ArgoverseDataSource('test', enable_cache=False),
        },
        strategy=DynamicSchedulingStrategy(exp_tag),
        num_classes=8,
        lr=0.001 / 64 * (1 or 2),
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    system.load_from_ckpt(
        Path(root_dir) / 'weights' / 'trained' / 'yolox_s_00_mAP=0.31668_3342393340_050132.ckpt'
    )
    res = run_sap(
        system, sap_factor=sap_factor, dataset_resize_ratio=2, dataset_fps=30,
        device_id=device_id,
    )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
