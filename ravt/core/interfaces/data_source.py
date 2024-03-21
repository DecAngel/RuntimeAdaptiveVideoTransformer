from dataclasses import dataclass
from enum import Enum
from typing import Tuple, NamedTuple, Optional, Generic, Dict, Any, Union, List

import numpy as np
from jaxtyping import UInt8, Float32, Int32
from torch.utils.data import Dataset


class NamedDict(Dict[str, Any]):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass
class ImageComponent(NamedDict):
    """Image component from a dataset.

    Attributes:
        image: a BGR image with HWC shape in 0-255
        original_size: the original HW size of the image
    """
    image: UInt8[np.ndarray, 'height width channel=3']
    original_size: Int32[np.ndarray, 'hw=2']


@dataclass
class BBoxComponent(NamedDict):
    """Bounding box component from a dataset.

    Attributes:
        coordinate: a xyxy array of bbox coordinates
        label: the category of the bbox
        probability: the probability of the bbox
        current_size: the current HW size of the bbox
        original_size: the original HW size of the bbox
    """
    coordinate: Float32[np.ndarray, 'objects xyxy=4']
    label: Int32[np.ndarray, 'objects']
    probability: Float32[np.ndarray, 'objects']
    current_size: Int32[np.ndarray, 'hw=2']
    original_size: Int32[np.ndarray, 'hw=2']


class DataComponent(Enum):
    IMAGE = 1
    BBOX = 2


class BaseDataSource(Dataset[Dict]):
    def prepare_data(self): raise NotImplementedError

    def get_image(self, seq_id: int, frame_id: int) -> ImageComponent: raise NotImplementedError

    def get_bbox(self, seq_id: int, frame_id: int) -> BBoxComponent: raise NotImplementedError

    def get_length(self) -> List[int]: raise NotImplementedError()

    def get_meta(self) -> Dict[str, Any]: raise NotImplementedError()

    def __getitem__(self, item: Tuple[int, int, DataComponent]) -> Dict:
        seq_id, frame_id, component = item
        if component == DataComponent.IMAGE:
            return self.get_image(seq_id, frame_id)
        elif component == DataComponent.BBOX:
            return self.get_bbox(seq_id, frame_id)
        else:
            raise ValueError(f"{component} is not supported!")
