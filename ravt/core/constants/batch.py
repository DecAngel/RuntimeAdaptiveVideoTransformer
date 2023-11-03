from typing import TypedDict, Union, List

import numpy as np
import torch
from jaxtyping import Shaped, Float32, Int32, UInt8


"""ID Type"""
IDTypeN = Int32[np.ndarray, '']
IDTypeT = Int32[torch.Tensor, '']


class _IDComponentDict(TypedDict, total=False):
    """Meta component of a batch."""
    image_id: IDTypeN
    seq_id: IDTypeN
    frame_id: IDTypeN
    clip_id: IDTypeN


class _IDBatchDict(TypedDict, total=False):
    image_id: Shaped[IDTypeT, '*batch_size time_m']
    seq_id: Shaped[IDTypeT, '*batch_size time_m']
    frame_id: Shaped[IDTypeT, '*batch_size time_m']
    clip_id: Shaped[IDTypeT, '*batch_size time_m']


"""Image Type"""
ImageTypeN = UInt8[np.ndarray, 'channel height width']
ImageSizeTypeN = Int32[np.ndarray, 'hw=2']
ImageTypeT = UInt8[torch.Tensor, 'channel height width']
ImageSizeTypeT = Int32[torch.Tensor, 'hw=2']


class ImageComponentDict(_IDComponentDict):
    """Image component of a batch.

    image: a BGR image with CHW shape in 0-255
    resize_ratio: Original HW size / current HW size
    """
    image: ImageTypeN
    original_size: ImageSizeTypeN


class ImageBatchDict(_IDBatchDict, total=False):
    image: Shaped[ImageTypeT, '*batch_size time_c']
    original_size: Shaped[ImageSizeTypeT, '*batch_size time_c']


"""BBox Type"""
BBoxCoordinateTypeN = Float32[np.ndarray, 'objects xyxy=4']
BBoxLabelTypeN = Int32[np.ndarray, 'objects']
BBoxProbabilityTypeN = Float32[np.ndarray, 'objects']
BBoxCoordinateTypeT = Float32[torch.Tensor, 'objects xyxy=4']
BBoxLabelTypeT = Int32[torch.Tensor, 'objects']
BBoxProbabilityTypeT = Float32[torch.Tensor, 'objects']


class BBoxComponentDict(_IDComponentDict):
    """BBox component of a batch.

    coordinate: the xyxy array of bbox coordinates
    label: the category of the bbox
    probability: the probability of the bbox
    """
    coordinate: BBoxCoordinateTypeN
    label: BBoxLabelTypeN
    probability: BBoxProbabilityTypeN


class BBoxBatchDict(_IDBatchDict, total=False):
    coordinate: Shaped[BBoxCoordinateTypeT, '*batch_size time_b']
    label: Shaped[BBoxLabelTypeT, '*batch_size time_b']
    probability: Shaped[BBoxProbabilityTypeT, 'batch_size time_b']


"""Batch Types"""
ComponentDict = Union[ImageComponentDict, BBoxComponentDict]


class BatchDict(TypedDict, total=False):
    image: ImageBatchDict
    bbox: BBoxBatchDict


class PredDict(TypedDict, total=False):
    bbox: BBoxBatchDict


class LossDict(TypedDict, total=False):
    loss: Float32[torch.Tensor, '']


class MetricDict(TypedDict, total=False):
    mAP: Float32[torch.Tensor, '']
    sAP: Float32[torch.Tensor, '']


"""Sample Dict"""


class SampleDict(TypedDict, total=False):
    seq_id: int
    frame_id: int
    image: List[int]
    bbox: List[int]
