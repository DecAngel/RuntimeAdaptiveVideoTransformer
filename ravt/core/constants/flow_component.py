import numpy as np
import torch
from jaxtyping import Float32, Int32

from .id_component import IDComponentNDict, IDComponentTDict


FlowTypeN = Float32[np.ndarray, 'channel=2 height width']
FlowSizeTypeN = Int32[np.ndarray, 'hw=2']
FlowTypeT = Float32[torch.Tensor, '*batch_size time channel=2 height width']
FlowSizeTypeT = Int32[torch.Tensor, '*batch_size time hw=2']


class FlowComponentNDict(IDComponentNDict, total=False):
    """Flow numpy component of a batch.

    **image**: a HW(YX) flow array with CHW shape in pixels

    **original_size**: Original HW size of flow
    """
    flow: FlowTypeN
    original_size: FlowSizeTypeN


class FlowComponentTDict(IDComponentTDict, total=False):
    """Flow tensor component of a batch.

    **image**: a HW(YX) flow array with CHW shape in pixels

    **original_size**: Original HW size of flow
    """
    flow: FlowTypeT
    original_size: FlowSizeTypeT
