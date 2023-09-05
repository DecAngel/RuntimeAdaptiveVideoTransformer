from typing import TypedDict, Optional, List, Union, Dict, Tuple, Literal, Sequence

import torch
import pytorch_lightning as pl
import kornia.augmentation as ka
from jaxtyping import Float, Int, UInt
from torch import nn

from ..blocks.augmentation import KorniaAugmentation, KorniaSequential, KorniaRandomChoice
from ..blocks.backbones import YOLOXPAFPNBackbone
from ..blocks.heads import YOLOXHead
from ..blocks.metrics import COCOEvalMAPMetric
from ..blocks.callbacks import EMACallback
from ..blocks.schedulers import StreamYOLOLR
from ravt.protocols.classes import BaseModel
from ravt.protocols.structures import ConfigTypes, InternalConfigs, BatchDict, LossDict, DatasetConfigsRequiredKeys


class YOLOXSystem(BaseModel):
    def __init__(
            self,
            # structural parameters
            depth_ratio: float = 0.33,
            width_ratio: float = 0.5,
            depthwise: bool = False,
            act: Literal['silu', 'relu', 'lrelu', 'sigmoid'] = 'silu',

            num_classes: int = 8,
            max_objs: int = 100,
            strides: Tuple[int, ...] = (8, 16, 32),
            in_channels: Tuple[int, ...] = (256, 512, 1024),

            # preprocess parameters

            # other parameters
            use_ema: bool = True,
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,
            lr: float = 0.001,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,

            **kwargs,
    ):
        super().__init__(metric=COCOEvalMAPMetric())
        self.save_hyperparameters(ignore='kwargs')
        self.backbone = YOLOXPAFPNBackbone(**self.hparams)
        self.head = YOLOXHead(**self.hparams)
        self.preprocess = KorniaAugmentation(
            train_aug=ka.RandomHorizontalFlip(),
            train_resize=KorniaRandomChoice(
                *[ka.Resize((h, w)) for h, w in zip(range(540, 661, 10), range(864, 1057, 16))]
            ),
            eval_aug=None,
            eval_resize=ka.Resize((600, 960)),
        )
        self.pretrained_dir = None

    @property
    def example_input_array(self) -> BatchDict:
        return {
            'meta': {
                'image_id': torch.ones(4, 3, dtype=torch.int32),
                'seq_id': torch.ones(4, 3, dtype=torch.int32),
                'frame_id': torch.ones(4, 3, dtype=torch.int32),
            },
            'image': {
                'image': torch.rand(4, 2, 3, 600, 960, dtype=torch.float32),
                'resize_ratio': torch.ones(4, 2, dtype=torch.float32),
            },
        }

    @property
    def required_keys_train(self) -> DatasetConfigsRequiredKeys:
        return {
            'interval': 3,
            'margin': 3,
            'components': {
                'meta': [1],
                'image': [0, 1],
                'bbox': [1, 2],
            }
        }

    @property
    def required_keys_eval(self) -> DatasetConfigsRequiredKeys:
        return {
            'interval': 1,
            'margin': 3,
            'components': {
                'meta': [1],
                'image': [0, 1],
            }
        }

    def forward_impl(self, batch: BatchDict) -> Union[BatchDict, LossDict]:
        batch = self.preprocess(batch)

        images: Float[torch.Tensor, 'B Ti C H W'] = batch['image']['image']
        coordinates: Float[torch.Tensor, 'B Tb O C'] = batch['bbox']['coordinate'] if 'bbox' in batch else None
        labels: UInt[torch.Tensor, 'B Tb O'] = batch['bbox']['label'] if 'bbox' in batch else None

        images *= 255.0
        _, image1 = images.unbind(1)
        coordinate1, _ = coordinates.unbind(1)
        label1, _ = labels.unbind(1)

        feature1 = self.backbone(image1)
        if self.training:
            loss_dict = self.head(
                feature1,
                gt_coordinates=coordinate1,
                gt_labels=label1,
                shape=image1.shape[-2:]
            )
            return loss_dict
        else:
            pred_dict = self.head(feature1, shape=image1.shape[-2:])
            return pred_dict

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'model':
            self.pretrained_dir = configs['environment']['weight_pretrained_dir']
        return configs

    def configure_optimizers(self):
        p_bias, p_norm, p_weight = [], [], []
        for k, v in self.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                p_bias.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                p_norm.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                p_weight.append(v.weight)  # apply decay
        optimizer = torch.optim.SGD(p_norm, lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True)
        optimizer.add_param_group({"params": p_weight, "weight_decay": self.hparams.weight_decay})
        optimizer.add_param_group({"params": p_bias})
        scheduler = StreamYOLOLR(optimizer, int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]

    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        if self.hparams.use_ema and self.trainer.training:
            return [EMACallback()]
        else:
            return []


def yolox_s(
        pretrained: bool = True,
        num_classes: int = 8,
        use_ema: bool = True,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
) -> YOLOXSystem:
    model = YOLOXSystem(
        num_classes=num_classes,
        use_ema=use_ema,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    if pretrained:
        def modify_pth(pth: Dict) -> Dict:
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

        file = model.pretrained_dir.joinpath('yolox_s.pth')
        if file.exists():
            model.load_state_dict(modify_pth(torch.load(str(file))), strict=False)
        else:
            raise FileNotFoundError(
                f'pretrained file yolox_s.pth '
                f'not found in {str(model.pretrained_dir)}!'
            )
    return model
