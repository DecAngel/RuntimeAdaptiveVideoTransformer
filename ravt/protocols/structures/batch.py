from typing import TypedDict

import torch
from jaxtyping import Float, Int, Shaped


"""Meta Type"""
MetaType = Int[torch.Tensor, '']


class MetaComponentDict(TypedDict, total=False):
    """Meta component of a batch."""
    image_id: MetaType
    seq_id: MetaType
    frame_id: MetaType


class MetaBatchDict(TypedDict, total=False):
    image_id: Shaped[MetaType, '*batch_size time_m']
    seq_id: Shaped[MetaType, '*batch_size time_m']
    frame_id: Shaped[MetaType, '*batch_size time_m']


"""Image Type"""
ImageType = Float[torch.Tensor, 'channel height width']
ImageSizeType = Int[torch.Tensor, 'hw=2']


class ImageComponentDict(MetaComponentDict):
    """Image component of a batch.

    image: a BGR image with CHW shape normalized to 0-1
    resize_ratio: Original HW size / current HW size
    """
    image: ImageType
    original_size: ImageSizeType


class ImageBatchDict(MetaBatchDict, total=False):
    image: Shaped[ImageType, '*batch_size time_c']
    original_size: Shaped[ImageSizeType, '*batch_size time_c']


"""BBox Type"""
BBoxCoordinateType = Float[torch.Tensor, 'objects xyxy=4']
BBoxLabelType = Int[torch.Tensor, 'objects']
BBoxProbabilityType = Float[torch.Tensor, 'objects']


class BBoxComponentDict(MetaComponentDict):
    """BBox component of a batch.

    coordinate: a xyxy array of bbox coordinates
    label: the category of the bbox
    """
    coordinate: BBoxCoordinateType
    label: BBoxLabelType


class BBoxBatchDict(MetaBatchDict, total=False):
    coordinate: Shaped[BBoxCoordinateType, '*batch_size time_b']
    label: Shaped[BBoxLabelType, '*batch_size time_b']
    probability: Shaped[BBoxProbabilityType, 'batch_size time_b']


"""Batch Types"""


class ComponentDict(TypedDict, total=False):
    image: ImageComponentDict
    bbox: BBoxComponentDict


class BatchDict(TypedDict, total=False):
    image: ImageBatchDict
    bbox: BBoxBatchDict


class PredDict(TypedDict, total=False):
    bbox: BBoxBatchDict


class LossDict(TypedDict, total=False):
    loss: Float[torch.Tensor, '']


class MetricDict(TypedDict, total=False):
    mAP: Float[torch.Tensor, '']
    sAP: Float[torch.Tensor, '']
