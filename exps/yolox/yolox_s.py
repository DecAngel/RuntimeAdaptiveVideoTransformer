import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from typing import Optional

import torch
import fire

from ravt.data_sources import ArgoverseDataSource
from ravt.models import yolox_s
from ravt.launchers import TrainTestLauncher

torch.set_float32_matmul_precision('high')


def main(
        exp_tag: str, predict_num: int = 0,
        batch_size: Optional[int] = None, device_id: int = 0, visualize: bool = False, debug: bool = False
):
    """ Train and test yolox_s model on Argoverse-HD

    :param exp_tag: the tag for the experiment
    :param predict_num: predict offset for the model
    :param batch_size: batch size of the exp, set None to auto-detect
    :param device_id: the cuda device id to place the model on
    :param visualize: enable visualization
    :param debug: enable debug mode
    :return:
    """
    batch_size = 4 if debug else batch_size
    num_workers = 0 if debug else 8
    data_source = ArgoverseDataSource()
    model = yolox_s(
        predict_num=predict_num,
        num_classes=8,
        lr=0.001 / 64 * (batch_size or 2),
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    model.load_from_pth(Path(root_dir) / 'weights' / 'pretrained' / 'yolox_s.pth')
    launcher = TrainTestLauncher(
        system=model, data_source=data_source, exp_tag=exp_tag, max_epoch=15,
        batch_size=batch_size, num_workers=num_workers, device_ids=[device_id], debug=debug,
        callback_ema=True, callback_visualize=visualize,
    )
    train_res = launcher.train(resume=None)
    test_res = launcher.test(resume='best')
    print(train_res, test_res)


if __name__ == '__main__':
    fire.Fire(main)
