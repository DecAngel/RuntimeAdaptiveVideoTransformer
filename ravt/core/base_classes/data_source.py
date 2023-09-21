from typing import List

from ..constants import (
    ComponentDict,
    SubsetTypes, ComponentTypes
)
from ..utils.phase_init import PhaseInitMixin


class BaseDataSource(PhaseInitMixin):
    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict:
        raise NotImplementedError()

    def get_length(self, subset: SubsetTypes) -> List[int]:
        raise NotImplementedError()
