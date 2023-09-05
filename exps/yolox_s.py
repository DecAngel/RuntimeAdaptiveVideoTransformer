import torch

from ravt.data_modules.argoverse import ArgoverseDataset
from ravt.model_modules.yolox import yolox_s
from ravt.launchers import TrainTestLauncher

from base_configs import base_configs


torch.set_float32_matmul_precision('high')


def main():
    batch_size = 64
    data_module = ArgoverseDataset(
        batch_size=batch_size,
        num_workers=2,
    )
    model = yolox_s(
        pretrained=True,
        num_classes=8,
        use_ema=False,
        lr=0.01 / 64 * batch_size,
        momentum=0.9,
        weight_decay=5e-4,
        conf_thre=0.01,
        nms_thre=0.65,
    )
    launcher = TrainTestLauncher(model=model, dataset=data_module, configs=base_configs, device_ids=[1], debug=True)
    print(launcher.train())
    print(launcher.test())


if __name__ == '__main__':
    main()
