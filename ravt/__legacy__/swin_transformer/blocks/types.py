from typing import Tuple, TypedDict

import torch
from jaxtyping import Float, Int


IMAGE = Float[torch.Tensor, 'batch_size channels_rgb=3 height width']
PYRAMID = Tuple[Float[torch.Tensor, 'batch_size channels height width'], ...]
COORDINATE = Float[torch.Tensor, 'batch_size max_objs coords_xyxy=4']
PROBABILITY = Float[torch.Tensor, 'batch_size max_objs']
LABEL = Int[torch.Tensor, 'batch_size max_objs']
LOSS = Float[torch.Tensor, '']


class YOLOXPredDict(TypedDict):
    pred_coordinates: COORDINATE
    pred_probabilities: PROBABILITY
    pred_labels: LABEL


class YOLOXLossDict(TypedDict):
    loss: LOSS
