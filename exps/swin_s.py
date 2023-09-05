import torch
import imgaug.augmenters as iaa

from ravt.data_modules.argoverse import ArgoverseDataset
from ravt.model_modules.swin_transformer import swin_transformer_small_patch4_window7
from ravt.launchers import run_train_and_test

torch.set_float32_matmul_precision('high')


def main():
    data_module = ArgoverseDataset(
        train_transform=iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.CropAndPad(percent=(-0.1, 0.1)),
            iaa.ChangeColorspace(
                from_colorspace=iaa.ChangeColorspace.BGR,
                to_colorspace=iaa.ChangeColorspace.RGB,
            ),
            iaa.Resize({'height': 600, 'width': 960}),
        ]),
        val_transform=iaa.Resize({'height': 600, 'width': 960}),
        clip_image_length=1,
        clip_label_length=1,
        batch_size=16,
        num_workers=8,
    )
    model = swin_transformer_small_patch4_window7(
        pretrained=True,
        num_classes=8,
        lr=0.0001,
        gamma=0.98,
        conf_thre=0.6,
        nms_thre=0.45,
        frozen_stages=5,
    )
    run_train_and_test(model=model, data_module=data_module, device_ids=[1])


if __name__ == '__main__':
    main()
