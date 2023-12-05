from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Int32


IDTypeN = Int32[np.ndarray, '']
IDTypeT = Int32[torch.Tensor, '*batch_size time_m']


class IDComponentNDict(TypedDict, total=False):
    """ID numpy component of a batch.

    **clip_id**: the relative temporal position in a batch
    """
    clip_id: IDTypeN


class IDComponentTDict(TypedDict, total=False):
    """ID tensor component of a batch.

    **clip_id**: the relative temporal position in a batch
    """
    clip_id: IDTypeT
