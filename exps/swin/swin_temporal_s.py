import torch

from ravt.data_modules.argoverse import ArgoverseDataset
from ravt.model_modules.swin_transformer import swin_transformer_temporal_small_patch4_window7
from ravt.launchers import TrainTestLauncher

from base_configs import environment_configs


torch.set_float32_matmul_precision('high')


def main():
    debug = True
    batch_size = 4 if debug else 8
    data_module = ArgoverseDataset(
        batch_size=batch_size,
        num_workers=0 if debug else 0,
    )
    model = swin_transformer_temporal_small_patch4_window7(
        pretrained=True,
        predict_num=1,
        num_classes=8,
        lr=0.001 / 64 * batch_size,
        gamma=0.98,
        conf_thre=0.6,
        nms_thre=0.45,
        frozen_stages=5,
    )
    launcher = TrainTestLauncher(
        system=model, data_source=data_module, envs=environment_configs, device_ids=[1], debug=debug,
        callback_ema=False, callback_visualize=False,
    )
    print(launcher.train())
    print(launcher.test())


if __name__ == '__main__':
    main()
