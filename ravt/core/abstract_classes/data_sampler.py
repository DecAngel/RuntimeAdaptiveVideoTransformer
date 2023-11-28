from typing import List, Protocol

from ravt.core.constants import SubsetTypes, SampleDict


class BaseDataSampler(Protocol):
    def sample(self, subset: SubsetTypes, seq_lengths: List[int]) -> List[SampleDict]: ...
