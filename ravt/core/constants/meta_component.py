from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Int32

MetaTypeN = Int32[np.ndarray, '']
MetaTypeT = Int32[torch.Tensor, '*batch_size']


class MetaComponentNDict(TypedDict, total=False):
    """Meta numpy component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in <seq_id> sequence
    """
    image_id: MetaTypeN
    seq_id: MetaTypeN
    frame_id: MetaTypeN


class MetaComponentTDict(TypedDict, total=False):
    """Meta tensor component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in <seq_id> sequence
    """
    image_id: MetaTypeT
    seq_id: MetaTypeT
    frame_id: MetaTypeT
