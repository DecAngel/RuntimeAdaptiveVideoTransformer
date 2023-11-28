from pathlib import Path
from typing import List, Protocol

from ..constants import (
    ComponentDict,
    SubsetTypes, ComponentTypes
)


class BaseDataSource(Protocol):
    def get_component(
            self, subset: SubsetTypes, component: ComponentTypes, seq_id: int, frame_id: int
    ) -> ComponentDict: ...

    def get_length(self, subset: SubsetTypes) -> List[int]: ...

    def get_image_dir(self, subset: SubsetTypes) -> Path: ...

    def get_ann_file(self, subset: SubsetTypes) -> Path: ...
