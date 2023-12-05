import numpy as np
import torch
from jaxtyping import Int32, UInt8

from .id_component import IDComponentNDict, IDComponentTDict


ImageTypeN = UInt8[np.ndarray, 'channel=3 height width']
ImageSizeTypeN = Int32[np.ndarray, 'hw=2']
ImageTypeT = UInt8[torch.Tensor, '*batch_size time channel=3 height width']
ImageSizeTypeT = Int32[torch.Tensor, 'hw=2']


class ImageComponentNDict(IDComponentNDict, total=False):
    """Image numpy component of a batch.

    **image**: a BGR image with CHW shape in 0-255

    **original_size**: Original HW size of image
    """
    image: ImageTypeN
    original_size: ImageSizeTypeN


class ImageComponentTDict(IDComponentTDict, total=False):
    """Image tensor component of a batch.

    **image**: a BGR image with CHW shape in 0-255

    **original_size**: Original HW size of image
    """
    image: ImageTypeT
    original_size: ImageSizeTypeT
