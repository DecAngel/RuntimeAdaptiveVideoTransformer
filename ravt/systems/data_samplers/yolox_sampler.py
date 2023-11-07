import functools
import random
from typing import List, Optional

from ravt.core.base_classes import BaseDataSampler
from ravt.core.constants import SubsetTypes, SampleDict


class YOLOXDataSampler(BaseDataSampler):
    def __init__(
            self,
            interval: int,
            eval_image_clip: List[int],
            eval_bbox_clip: List[int],
            train_image_clip: Optional[List[List[int]]] = None,
            train_bbox_clip: Optional[List[List[int]]] = None,
    ):
        self.interval = interval
        self.eval_image_clip = eval_image_clip
        self.eval_bbox_clip = eval_bbox_clip
        self.train_image_clip = train_image_clip or [eval_image_clip]
        self.train_bbox_clip = train_bbox_clip or [eval_bbox_clip]

        eval_all = eval_image_clip + eval_bbox_clip
        train_all = functools.reduce(list.__add__, train_image_clip + train_bbox_clip)
        self.eval_margin = min(eval_all), max(eval_all)
        self.train_margin = min(train_all), max(train_all)

    def sample(self, subset: SubsetTypes, seq_lengths: List[int]) -> List[SampleDict]:
        if subset == 'train':
            return functools.reduce(list.__add__, [
                [
                    {
                        'seq_id': seq_id,
                        'frame_id': frame_id,
                        'image': self.train_image_clip[rand_id],
                        'bbox': self.train_bbox_clip[rand_id],
                    }
                    for frame_id, rand_id in zip(
                        range(-self.train_margin[0], seq_len-self.train_margin[1]),
                        random.choices(range(len(self.train_image_clip)), k=seq_len-self.train_margin[1]+self.train_margin[0]),
                    )
                ]
                for seq_id, seq_len in enumerate(seq_lengths)
            ])
        else:
            return functools.reduce(list.__add__, [
                [
                    {
                        'seq_id': seq_id,
                        'frame_id': frame_id,
                        'image': self.eval_image_clip,
                        'bbox': self.eval_bbox_clip,
                    }
                    for frame_id in range(-self.eval_margin[0], seq_len-self.eval_margin[1])
                ]
                for seq_id, seq_len in enumerate(seq_lengths)
            ])
