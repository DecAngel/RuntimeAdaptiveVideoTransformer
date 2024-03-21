from typing import NamedTuple, Union

import numpy as np
import torch
from jaxtyping import UInt8, Float32, Int32


ArrayType = Union[np.ndarray, torch.Tensor]


class ImageComponent(NamedTuple):
    """Image component from a dataset.

    Attributes:
        image: a BGR image with HWC shape in 0-255
        original_size: the original HW size of the image
    """
    image: UInt8[ArrayType, '*batch_time height width channel=3']
    original_size: Int32[ArrayType, '*batch_time hw=2']


class BBoxComponent(NamedTuple):
    """Bounding box component from a dataset.

    Attributes:
        coordinate: a xyxy array of bbox coordinates
        label: the category of the bbox
        probability: the probability of the bbox
        current_size: the current HW size of the bbox
        original_size: the original HW size of the bbox
    """
    coordinate: Float32[ArrayType, '*batch_time objects xyxy=4']
    label: Int32[ArrayType, '*batch_time objects']
    probability: Float32[ArrayType, '*batch_time objects']
    current_size: Int32[ArrayType, '*batch_time hw=2']
    original_size: Int32[ArrayType, '*batch_time hw=2']
