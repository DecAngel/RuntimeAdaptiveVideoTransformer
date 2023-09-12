from pathlib import Path
from typing import Callable, Dict

import torch
from ravt.utils.lightning_logger import ravt_logger as logger


def get_pth_adapter(filename: str) -> Callable[[Path], Dict]:
    def pth_adapter(pretrained_dir: Path) -> Dict:
        file = pretrained_dir.joinpath(filename)
        if file.exists():
            pth = torch.load(str(file))
            logger.info(f'Load pretrained file {filename}')
        else:
            raise FileNotFoundError(
                f'pretrained file {filename} not found in {str(pretrained_dir)}!'
            )
        new_ckpt = {}
        for k, v in pth['model'].items():
            if 'head' in k:
                continue
            k = k.replace('lateral_conv0', 'down_conv_5_4')
            k = k.replace('C3_p4', 'down_csp_4_4')
            k = k.replace('reduce_conv1', 'down_conv_4_3')
            k = k.replace('C3_p3', 'down_csp_3_3')
            k = k.replace('bu_conv2', 'up_conv_3_3')
            k = k.replace('C3_n3', 'up_csp_3_4')
            k = k.replace('bu_conv1', 'up_conv_4_4')
            k = k.replace('C3_n4', 'up_csp_4_5')
            k = k.replace('backbone.jian2', 'neck.convs.0')
            k = k.replace('backbone.jian1', 'neck.convs.1')
            k = k.replace('backbone.jian0', 'neck.convs.2')
            new_ckpt[k] = v
        return new_ckpt
    return pth_adapter
