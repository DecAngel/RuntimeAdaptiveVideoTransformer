from typing import TypedDict

import torch
from jaxtyping import Float, Int, UInt, Shaped


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
ImageRatioType = Float[torch.Tensor, 'hw=2']


class ImageComponentDict(TypedDict):
    """Image component of a batch.

    image: a BGR image with CHW shape normalized to 0-1
    resize_ratio: Original HW size / current HW size
    """
    image: ImageType
    resize_ratio: ImageRatioType


class ImageBatchDict(TypedDict, total=False):
    image: Shaped[ImageType, '*batch_size time_c']
    resize_ratio: Shaped[ImageRatioType, '*batch_size']


"""BBox Type"""
BBoxCoordinateType = Float[torch.Tensor, 'objects xyxy=4']
BBoxLabelType = UInt[torch.Tensor, 'objects']
BBoxProbabilityType = Float[torch.Tensor, 'objects']


class BBoxComponentDict(TypedDict):
    """BBox component of a batch.

    coordinate: a xyxy array of bbox coordinates
    label: the category of the bbox
    """
    coordinate: BBoxCoordinateType
    label: BBoxLabelType


class BBoxBatchDict(TypedDict, total=False):
    coordinate: Shaped[BBoxCoordinateType, '*batch_size time_b']
    label: Shaped[BBoxLabelType, '*batch_size time_b']
    probability: Shaped[BBoxProbabilityType, 'batch_size time_b']


"""Batch Types"""


class ComponentDict(TypedDict, total=False):
    meta: MetaComponentDict
    image: ImageComponentDict
    bbox: BBoxComponentDict


class BatchDict(TypedDict, total=False):
    meta: MetaBatchDict
    image: ImageBatchDict
    bbox: BBoxBatchDict


class LossDict(TypedDict, total=False):
    loss: Float[torch.Tensor, '']


class MetricDict(TypedDict, total=False):
    mAP: Float[torch.Tensor, '']
