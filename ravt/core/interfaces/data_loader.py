from dataclasses import dataclass
from enum import Enum
from typing import Tuple, NamedTuple, Optional, Generic, Dict, Any, Union, List

import numpy as np
from jaxtyping import UInt8, Float32, Int32
from torch.utils.data import Dataset, DataLoader

from .data_source import BaseDataSource


class BaseDataLoader(DataLoader[Dict]):
    def __init__(self, data_source: BaseDataSource, batch_size: int)