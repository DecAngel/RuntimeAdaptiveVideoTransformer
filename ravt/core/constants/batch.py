from typing import TypedDict, Union, List

import numpy as np
import torch
from jaxtyping import Float32, Int32, UInt8


# ID Type
IDTypeN = Int32[np.ndarray, '']
IDTypeT = Int32[torch.Tensor, '*batch_size time_m']


class _IDComponentNDict(TypedDict, total=False):
    """ID numpy component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in a sequence

    **clip_id**: the relative temporal position in a batch
    """
    image_id: IDTypeN
    seq_id: IDTypeN
    frame_id: IDTypeN
    clip_id: IDTypeN


class _IDComponentTDict(TypedDict, total=False):
    """ID tensor component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in a sequence

    **clip_id**: the relative temporal position in a batch
    """
    image_id: IDTypeT
    seq_id: IDTypeT
    frame_id: IDTypeT
    clip_id: IDTypeT


# Image Type
ImageTypeN = UInt8[np.ndarray, 'channel=3 height width']
ImageSizeTypeN = Int32[np.ndarray, 'hw=2']
ImageTypeT = UInt8[torch.Tensor, '*batch_size time channel=3 height width']
ImageSizeTypeT = Int32[torch.Tensor, 'hw=2']


class ImageComponentNDict(_IDComponentNDict, total=False):
    """Image numpy component of a batch.

    **image**: a BGR image with CHW shape in 0-255

    **original_size**: Original HW size of image
    """
    image: ImageTypeN
    original_size: ImageSizeTypeN


class ImageComponentTDict(_IDComponentTDict, total=False):
    """Image tensor component of a batch.

    **image**: a BGR image with CHW shape in 0-255

    **original_size**: Original HW size of image
    """
    image: ImageTypeT
    original_size: ImageSizeTypeT


# BBox Type
BBoxCoordinateTypeN = Float32[np.ndarray, 'objects xyxy=4']
BBoxLabelTypeN = Int32[np.ndarray, 'objects']
BBoxProbabilityTypeN = Float32[np.ndarray, 'objects']
BBoxCoordinateTypeT = Float32[torch.Tensor, '*batch_size time objects xyxy=4']
BBoxLabelTypeT = Int32[torch.Tensor, '*batch_size time objects']
BBoxProbabilityTypeT = Float32[torch.Tensor, '*batch_size time objects']


class BBoxComponentNDict(_IDComponentNDict, total=False):
    """BBox numpy component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox
    """
    coordinate: BBoxCoordinateTypeN
    label: BBoxLabelTypeN
    probability: BBoxProbabilityTypeN


class BBoxComponentTDict(_IDComponentTDict, total=False):
    """BBox tensor component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox
    """
    coordinate: BBoxCoordinateTypeT
    label: BBoxLabelTypeT
    probability: BBoxProbabilityTypeT


# Flow Types
FlowTypeN = Float32[np.ndarray, 'channel=2 height width']
FlowSizeTypeN = Int32[np.ndarray, 'hw=2']
FlowTypeT = Float32[torch.Tensor, '*batch_size time channel=2 height width']
FlowSizeTypeT = Int32[torch.Tensor, '*batch_size time hw=2']


class FlowComponentNDict(_IDComponentNDict, total=False):
    """Flow numpy component of a batch.

    **image**: a HW(YX) flow array with CHW shape in pixels

    **original_size**: Original HW size of flow
    """
    flow: FlowTypeN
    original_size: FlowSizeTypeN


class FlowComponentTDict(_IDComponentTDict, total=False):
    """Flow tensor component of a batch.

    **image**: a HW(YX) flow array with CHW shape in pixels

    **original_size**: Original HW size of flow
    """
    image: FlowTypeT
    original_size: FlowSizeTypeT


# Visualization Types
class VisualizationComponentTDict(_IDComponentTDict, total=False):
    image: ImageTypeT
    bbox_coordinate: BBoxCoordinateTypeT
    bbox_label: BBoxLabelTypeT
    bbox_probability: BBoxProbabilityTypeT
    flow: FlowTypeT


# Batch Types
ComponentDict = Union[ImageComponentNDict, BBoxComponentNDict, FlowComponentNDict]


class BatchDict(TypedDict, total=False):
    image: ImageComponentTDict
    bbox: BBoxComponentTDict
    flow: FlowComponentTDict


class PredDict(TypedDict, total=False):
    bbox: BBoxComponentTDict
    flow: FlowComponentTDict


class VisualizationDict(TypedDict, total=False):
    visualizations: List[VisualizationComponentTDict]


class LossDict(TypedDict, total=False):
    loss: Float32[torch.Tensor, '']


class MetricDict(TypedDict, total=False):
    mAP: Float32[torch.Tensor, '']
    sAP: Float32[torch.Tensor, '']


class OutputDict(PredDict, VisualizationDict, LossDict):
    pass


# Sample Dict
class SampleDict(TypedDict, total=False):
    seq_id: int
    frame_id: int
    image: List[int]
    bbox: List[int]
