from typing import List

from ravt.core.constants import SubsetTypes, SampleDict


class BaseDataSampler:
    def sample(self, subset: SubsetTypes, seq_lengths: List[int]) -> List[SampleDict]:
        raise NotImplementedError()
