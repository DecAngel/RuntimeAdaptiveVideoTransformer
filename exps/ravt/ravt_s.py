import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parents[2].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from typing import Tuple, Optional

import torch
import fire

from ravt.data_sources import ArgoverseDataSource
from ravt.models import ravt_small_patch4_window7
from ravt.launchers import TrainTestLauncher

torch.set_float32_matmul_precision('high')


def main(exp_tag: str, predict_nums: Tuple[int, ...] = (0, 1, 2), batch_size: Optional[int] = None, device_id: int = 0, visualize: bool = False, debug: bool = False):
    """ Train and test ravt_s model on Argoverse-HD

        :param batch_size:
        :param exp_tag: the tag for the experiment
        :param predict_nums: predict offsets for the model
        :param device_id: the cuda device id to place the model on
        :param visualize: enable visualization
        :param debug: enable debug mode
        :return:
        """
    batch_size = 4 if debug else batch_size
    num_workers = 0 if debug else 8
    data_source = ArgoverseDataSource()
    model = ravt_small_patch4_window7(
        predict_nums=predict_nums,
        num_classes=8,
        lr=0.001 / 64 * (batch_size or 2),
        gamma=0.98,
        weight_decay=0.05,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        conf_thre=0.01,
        nms_thre=0.65,
        frozen_stages=1,
    )
    model.load_from_pth(Path(root_dir) / 'weights' / 'pretrained' / 'cascade_mask_rcnn_swin_small_patch4_window7.pth')
    launcher = TrainTestLauncher(
        system=model, data_source=data_source, exp_tag=exp_tag, max_epoch=20,
        batch_size=batch_size, num_workers=num_workers, device_ids=[device_id], debug=debug,
        callback_ema=False, callback_visualize=visualize,
    )
    train_res = launcher.train(resume=None)
    test_res = launcher.test(resume='best')
    print(train_res, test_res)


if __name__ == '__main__':
    fire.Fire(main)

