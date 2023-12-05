from typing import List, Tuple, Dict

from ..constants import ComponentLiteral


class BaseDataSampler:
    def sample(
            self, train: bool, seq_lengths: List[int]
    ) -> List[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]: raise NotImplementedError()
