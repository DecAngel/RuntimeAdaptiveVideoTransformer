from pathlib import Path
from typing import List, Tuple, Dict

from torch.utils.data import Dataset, default_collate

from ..constants import ComponentNDict, BatchNDict, ComponentLiteral


class BaseDataSource(Dataset):
    img_dir: Path
    ann_file: Path

    def get_component(
            self, seq_id: int, frame_id: int, component: ComponentLiteral,
    ) -> ComponentNDict: raise NotImplementedError()

    def get_length(self) -> List[int]: raise NotImplementedError()

    def __getitem__(self, item: Tuple[int, int, Dict[ComponentLiteral, List[int]]]) -> BatchNDict:
        seq_id, frame_id, components = item
        batch: BatchNDict = self.get_component(seq_id, frame_id, 'meta')
        for c, clip_ids in components.items():
            res = []
            for i in clip_ids:
                res.append(self.get_component(seq_id, frame_id+i, c))
            batch[c] = default_collate(res)
        return batch
