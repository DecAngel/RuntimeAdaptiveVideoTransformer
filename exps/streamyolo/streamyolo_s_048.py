import torch

from ravt.data_sources import ArgoverseDataSource
from ravt.models import streamyolo_s
from ravt.launchers import TrainTestLauncher

torch.set_float32_matmul_precision('high')


def main():
    debug = False
    batch_size = 4 if debug else 16
    num_workers = 0 if debug else 8
    data_source = ArgoverseDataSource()
    model = streamyolo_s(
        pretrained=True,
        predict_num=4,
        neck_type='dfp',
        num_classes=8,
        lr=0.01 / 64 * batch_size,
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    launcher = TrainTestLauncher(
        system=model, data_source=data_source,
        batch_size=batch_size, num_workers=num_workers, device_ids=[1], debug=debug,
        callback_ema=True, callback_visualize=False,
    )
    print(launcher.train())
    print(launcher.test())


if __name__ == '__main__':
    main()
