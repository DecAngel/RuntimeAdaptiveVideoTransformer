import torch

from ravt.data_sources import ArgoverseDataSource
from ravt.models import swin_transformer_small_patch4_window7
from ravt.launchers import TrainTestLauncher


torch.set_float32_matmul_precision('high')


def main():
    debug = False
    batch_size = 4 if debug else 16
    num_workers = 0 if debug else 8
    data_source = ArgoverseDataSource()
    model = swin_transformer_small_patch4_window7(
        pretrained=True,
        num_classes=8,
        predict_num=0,
        lr=0.0002,
        gamma=0.98,
        conf_thre=0.6,
        nms_thre=0.45,
        frozen_stages=3,
    )
    launcher = TrainTestLauncher(
        system=model, data_source=data_source,
        batch_size=batch_size, num_workers=num_workers, device_ids=[1], debug=debug,
        callback_ema=False, callback_visualize=False,
    )
    print(launcher.train())
    print(launcher.test())


if __name__ == '__main__':
    main()
