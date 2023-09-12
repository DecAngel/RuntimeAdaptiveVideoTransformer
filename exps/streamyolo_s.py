import torch

from ravt.data_modules.argoverse import ArgoverseDataset
from ravt.model_modules.yolox import streamyolo_s
from ravt.launchers import TrainTestLauncher

from base_configs import environment_configs


torch.set_float32_matmul_precision('high')


def main():
    debug = False
    batch_size = 4 if debug else 16
    data_module = ArgoverseDataset(
        batch_size=batch_size,
        num_workers=0 if debug else 0,
    )
    model = streamyolo_s(
        pretrained=True,
        num_classes=8,
        lr=0.01 / 64 * batch_size,
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    launcher = TrainTestLauncher(
        model=model, dataset=data_module, envs=environment_configs, device_ids=[0], debug=debug,
        callback_ema=True, callback_visualize=False,
    )
    print(launcher.train())
    print(launcher.test())


if __name__ == '__main__':
    main()
