import numpy as np
import torch
from jaxtyping import Float32, Int32


from .id_component import IDComponentNDict, IDComponentTDict


BBoxCoordinateTypeN = Float32[np.ndarray, 'objects xyxy=4']
BBoxLabelTypeN = Int32[np.ndarray, 'objects']
BBoxProbabilityTypeN = Float32[np.ndarray, 'objects']
BBoxSizeTypeN = Int32[np.ndarray, 'hw=2']
BBoxCoordinateTypeT = Float32[torch.Tensor, '*batch_size time objects xyxy=4']
BBoxLabelTypeT = Int32[torch.Tensor, '*batch_size time objects']
BBoxProbabilityTypeT = Float32[torch.Tensor, '*batch_size time objects']
BBoxSizeTypeT = Int32[torch.Tensor, '*batch_size time hw=2']


class BBoxComponentNDict(IDComponentNDict, total=False):
    """BBox numpy component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox
    """
    coordinate: BBoxCoordinateTypeN
    label: BBoxLabelTypeN
    probability: BBoxProbabilityTypeN
    original_size: BBoxSizeTypeN


class BBoxComponentTDict(IDComponentTDict, total=False):
    """BBox tensor component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox
    """
    coordinate: BBoxCoordinateTypeT
    label: BBoxLabelTypeT
    probability: BBoxProbabilityTypeT
    original_size: BBoxSizeTypeT
