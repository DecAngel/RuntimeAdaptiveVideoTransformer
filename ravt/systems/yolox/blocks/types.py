from typing import Tuple, TypedDict, List, Union, Optional

import torch
from torch import nn
from jaxtyping import Float, Int


IMAGE = Float[torch.Tensor, 'batch_size time channels_rgb=3 height width']
PYRAMID = Tuple[Float[torch.Tensor, 'batch_size time channels height width'], ...]
COORDINATE = Float[torch.Tensor, 'batch_size time max_objs coords_xyxy=4']
PROBABILITY = Float[torch.Tensor, 'batch_size time max_objs']
LABEL = Int[torch.Tensor, 'batch_size time max_objs']
LOSS = Float[torch.Tensor, '']

TIME = Float[torch.Tensor, 'batch_size time']


class YOLOXPredDict(TypedDict):
    pred_coordinates: COORDINATE
    pred_probabilities: PROBABILITY
    pred_labels: LABEL


class YOLOXLossDict(TypedDict):
    loss: LOSS


class BaseBackbone(nn.Module):
    def forward(self, image: IMAGE) -> PYRAMID:
        raise NotImplementedError()


class BaseNeck(nn.Module):
    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> PYRAMID:
        raise NotImplementedError()


class BaseHead(nn.Module):
    def forward(
            self,
            features: PYRAMID,
            gt_coordinates: Optional[COORDINATE] = None,
            gt_labels: Optional[LABEL] = None,
            shape: Optional[Tuple[int, int]] = (600, 960),
    ) -> Union[YOLOXPredDict, YOLOXLossDict]:
        raise NotImplementedError()
