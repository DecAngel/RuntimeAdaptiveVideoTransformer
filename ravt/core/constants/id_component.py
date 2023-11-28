from typing import TypedDict, Union, List

import numpy as np
import torch
from jaxtyping import Float32, Int32, UInt8


IDTypeN = Int32[np.ndarray, '']
IDTypeT = Int32[torch.Tensor, '*batch_size time_m']


class IDComponentNDict(TypedDict, total=False):
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


class IDComponentTDict(TypedDict, total=False):
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
