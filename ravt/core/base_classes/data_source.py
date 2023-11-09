from pathlib import Path
from typing import List

from ..constants import (
    ComponentDict,
    SubsetTypes, ComponentTypes
)


class BaseDataSource:
    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict:
        raise NotImplementedError()

    def get_length(self, subset: SubsetTypes) -> List[int]:
        raise NotImplementedError()

    def get_image_dir(self, subset: SubsetTypes) -> Path:
        raise NotImplementedError()

    def get_ann_file(self, subset: SubsetTypes) -> Path:
        raise NotImplementedError()
