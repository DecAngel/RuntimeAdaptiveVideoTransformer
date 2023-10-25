import os
import sys
from pathlib import Path

import torch

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from ravt.sap_strategies import NormalStrategy, DynamicSchedulingStrategy, DadeSchedulingStrategy
from ravt.data_sources import ArgoverseDataSource
from ravt.models import yolox_s, streamyolo_s, msca_s
from ravt.launchers import SAPLauncher

torch.set_float32_matmul_precision('high')


def main():
    device_id = 1
    sap_factor = 2.0

    data_source = ArgoverseDataSource()
    model = streamyolo_s(
        predict_num=2,
        neck_type='dfp',
        num_classes=8,
        lr=0.001 / 64,
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    model.load_state_dict(
        torch.load(
            'weights/trained/streamyolo_s_012_mAP=0.28381_231232_c6ed78_3588138430.ckpt',
            map_location=torch.device(f'cuda:{device_id}')
        )['state_dict'],
        strict=True,
    )
    strategy = DadeSchedulingStrategy()
    launcher = SAPLauncher(
        model, data_source, strategy, device_id=device_id, debug=False
    )
    print(f'sAP:{launcher.test_sap(sap_factor)}')


if __name__ == '__main__':
    main()
