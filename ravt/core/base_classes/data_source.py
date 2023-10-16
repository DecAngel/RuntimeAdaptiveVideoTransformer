from typing import List, Tuple

from ..constants import (
    ComponentDict,
    SubsetTypes, ComponentTypes
)
from ..utils.phase_init import PhaseInitMixin, PhaseTypes, AllConfigs


class BaseDataSource(PhaseInitMixin):
    def __init__(self, original_size_hw: Tuple[int, int]):
        super().__init__()
        self.original_size = original_size_hw

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'environment':
            configs['internal']['original_size'] = self.original_size
        return configs

    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict:
        raise NotImplementedError()

    def get_length(self, subset: SubsetTypes) -> List[int]:
        raise NotImplementedError()
