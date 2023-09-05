import contextlib
from typing import TypedDict, Optional, List, Union, Dict, Tuple

import torch
import typeguard
import pytorch_lightning as pl
from jaxtyping import Float, Int, UInt
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ..common.backbones import SwinTransformerBackbone
from ..common.heads import YOLOXHead
from ..common.metrics import COCOEvalMAPMetric
from ravt.configs import weight_pretrained_dir


class SwinTransformerSystem(pl.LightningModule):
    class OutputPredTypedDict(TypedDict):
        pred_coordinates: Float[torch.Tensor, 'batch_size max_objs coords_xyxy=4']
        pred_probabilities: Float[torch.Tensor, 'batch_size max_objs']
        pred_labels: Int[torch.Tensor, 'batch_size max_objs']

    class OutputLossTypedDict(TypedDict):
        loss: Float[torch.Tensor, '']

    def __init__(
            self,
            # structural parameters
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 18, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: int = 7,
            ape: bool = False,
            drop_path_rate: float = 0.2,
            patch_norm: bool = True,
            use_checkpoint: bool = False,
            frozen_stages: int = -1,

            num_classes: int = 8,
            max_objs: int = 100,
            strides: Tuple[int, ...] = (4, 8, 16, 32),
            in_channels: Tuple[int, ...] = (96, 192, 384, 768),

            # preprocess parameters
            norm_mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
            norm_std: Tuple[float, ...] = (58.395, 57.12, 57.375),

            # other parameters
            conf_thre: float = 0.01,
            nms_thre: float = 0.65,
            lr: float = 0.001,
            gamma: float = 0.99,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore='kwargs')

        self.backbone = SwinTransformerBackbone(**self.hparams)
        self.head = YOLOXHead(width_ratio=1.0, **self.hparams)
        self.metric = COCOEvalMAPMetric()
        self.register_buffer('norm_mean', torch.tensor(self.hparams.norm_mean))
        self.register_buffer('norm_std', torch.tensor(self.hparams.norm_std))
        self.example_input_array = {
            'image_ids': torch.arange(0, 4),
            'resize_ratios': torch.ones(4, 2),
            'images': [torch.randint(0, 255, (4, 600, 960, 3), dtype=torch.uint8)]
        }

    def setup(self, stage: str) -> None:
        self.metric.setup(self.trainer, stage)

    def preprocess(
            self, images: UInt[torch.Tensor, 'batch_size height width channels_rgb=3']
    ) -> TypedDict('PreprocessedTypedDict', {
        'images': Float[torch.Tensor, 'batch_size channels=3 height width'],
    }):
        augmented_images = (images - self.norm_mean) / self.norm_std
        augmented_images = torch.einsum('BHWC->BCHW', augmented_images)
        return {'images': augmented_images}

    @typeguard.typechecked()
    def forward(
            self,
            images: List[UInt[torch.Tensor, 'batch_size height width channels_rgb=3']],
            gt_coordinates: Optional[List[Float[torch.Tensor, 'batch_size max_objs coords_xyxy=4']]] = None,
            gt_labels: Optional[List[Int[torch.Tensor, 'batch_size max_objs']]] = None,
            **kwargs,
    ) -> Union[OutputPredTypedDict, OutputLossTypedDict]:
        context = contextlib.nullcontext() if self.training else torch.no_grad()
        with context:
            image_center = images[-1]
            coordinates = None if gt_coordinates is None else gt_coordinates[0]
            labels = None if gt_labels is None else gt_labels[0]

            image_center = self.preprocess(image_center)['images']
            features = self.backbone(image_center)['features']

            if self.training:
                loss_dict = self.head(
                    features,
                    gt_coordinates=coordinates,
                    gt_labels=labels,
                    shape=image_center.shape[-2:]
                )
                return loss_dict
            else:
                pred_dict = self.head(features, shape=image_center.shape[-2:])
                return pred_dict

    def log_common(
            self,
            image_ids: Optional[Int[torch.Tensor, 'batch_size']] = None,
            seq_ids: Optional[Int[torch.Tensor, 'batch_size']] = None,
            frame_ids: Optional[Int[torch.Tensor, 'batch_size']] = None,
            **kwargs,
    ):
        if not self.trainer.sanity_checking:
            if image_ids is not None:
                self.log('image_id', image_ids[0].float(), batch_size=image_ids.shape[0], on_step=True, on_epoch=False, prog_bar=True)
            if seq_ids is not None:
                self.log('seq_id', seq_ids[0].float(), on_step=True)
            if frame_ids is not None:
                self.log('frame_id', seq_ids[0].float(), on_step=True)

    def training_step(self, batch: Dict, *args, **kwargs) -> OutputLossTypedDict:
        loss_dict = self.forward(**batch)
        self.log_common(**batch)
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=True, prog_bar=True)
        self.log('loss', loss_dict['loss'], on_step=True, prog_bar=True)
        return loss_dict

    def validation_step(self, batch: Dict, *args, **kwargs) -> None:
        pred_dict = self.forward(**batch)
        self.log_common(**batch)
        if not self.trainer.sanity_checking:
            self.metric.update(**batch, **pred_dict)

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            for k, v in self.metric.compute().items():
                if k.lower() == 'map':
                    self.log('mAP', v, prog_bar=True, sync_dist=True)
                else:
                    self.log(k, v)
            self.metric.reset()

    def test_step(self, batch: Dict, *args, **kwargs) -> None:
        pred_dict = self.forward(**batch)
        if not self.trainer.sanity_checking:
            self.metric.update(**batch, **pred_dict)

    def on_test_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            for k, v in self.metric.compute().items():
                if k.lower() == 'map':
                    self.log('mAP', v, prog_bar=True, sync_dist=True)
                else:
                    self.log(k, v)
            self.metric.reset()

    def predict_step(self, batch: Dict, *args, **kwargs) -> OutputPredTypedDict:
        return self.forward(**batch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.05)
        scheduler = StepLR(optimizer, 1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


def swin_transformer_small_patch4_window7(
        pretrained: bool = True,
        num_classes: int = 8,
        conf_thre: float = 0.01,
        nms_thre: float = 0.65,
        lr: float = 0.001,
        gamma: float = 0.99,
        **kwargs,
) -> SwinTransformerSystem:
    model = SwinTransformerSystem(
        num_classes=num_classes,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        lr=lr,
        gamma=gamma,
        **kwargs
    )
    if pretrained:
        def modify_pth(pth: Dict) -> Dict:
            return pth['state_dict']

        file = weight_pretrained_dir.joinpath('cascade_mask_rcnn_swin_small_patch4_window7.pth')
        if file.exists():
            model.load_state_dict(modify_pth(torch.load(str(file))), strict=False)
        else:
            raise FileNotFoundError(
                f'pretrained file cascade_mask_rcnn_swin_small_patch4_window7.pth '
                f'not found in {str(weight_pretrained_dir)}!'
            )
    return model
