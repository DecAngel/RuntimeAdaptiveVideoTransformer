from typing import TypedDict, Union

import torch
from jaxtyping import Float32

from .image_component import ImageComponentNDict, ImageComponentTDict
from .bbox_component import BBoxComponentNDict, BBoxComponentTDict
from .flow_component import FlowComponentNDict, FlowComponentTDict
from .feature_component import FeatureComponentNDict, FeatureComponentTDict
from .meta_component import MetaComponentNDict, MetaComponentTDict


class BatchNDict(MetaComponentNDict, total=False):
    image: ImageComponentNDict
    bbox: BBoxComponentNDict
    flow: FlowComponentNDict
    feature: FeatureComponentNDict


class BatchTDict(MetaComponentTDict, total=False):
    image: ImageComponentTDict
    bbox: BBoxComponentTDict
    flow: FlowComponentTDict
    feature: FeatureComponentTDict


class LossDict(TypedDict, total=False):
    loss: Float32[torch.Tensor, '']


class MetricDict(TypedDict, total=False):
    mAP: Float32[torch.Tensor, '']
    sAP: Float32[torch.Tensor, '']


ComponentNDict = Union[
    MetaComponentNDict, ImageComponentNDict, BBoxComponentNDict, FlowComponentNDict, FeatureComponentNDict,
    BatchNDict,
]
