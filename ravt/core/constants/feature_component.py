import numpy as np
import torch
from jaxtyping import Float32, Int32

from .id_component import IDComponentNDict, IDComponentTDict


FeatureTypeN = Float32[np.ndarray, 'channel height width']
FeatureSizeTypeN = Int32[np.ndarray, 'hw=2']
FeatureTypeT = Float32[torch.Tensor, '*batch_size time channel height width']
FeatureSizeTypeT = Int32[torch.Tensor, '*batch_size time hw=2']


class FeatureComponentNDict(IDComponentNDict, total=False):
    """Feature numpy component of a batch.

    **image**: a feature array with CHW shape in pixels

    **original_size**: Original HW size of flow
    """
    feature: FeatureTypeN
    original_size: FeatureSizeTypeN


class FeatureComponentTDict(IDComponentTDict, total=False):
    """Feature tensor component of a batch.

    **image**: a feature array with CHW shape in pixels

    **original_size**: Original HW size of flow
    """
    feature: FeatureTypeT
    original_size: FeatureSizeTypeT
