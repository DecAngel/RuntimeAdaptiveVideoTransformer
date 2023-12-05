import functools
import random
from typing import List, Optional, Tuple, Dict

from ravt.core.base_classes import BaseDataSampler
from ravt.core.constants import ComponentLiteral


class YOLOXDataSampler(BaseDataSampler):
    def __init__(
            self,
            interval: int,
            eval_image_clip: List[int],
            eval_bbox_clip: List[int],
            train_image_clip: Optional[List[List[int]]] = None,
            train_bbox_clip: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        self.interval = interval
        self.eval_image_clip = eval_image_clip
        self.eval_bbox_clip = eval_bbox_clip
        self.train_image_clip = train_image_clip or [eval_image_clip]
        self.train_bbox_clip = train_bbox_clip or [eval_bbox_clip]

        eval_all = eval_image_clip + eval_bbox_clip
        train_all = functools.reduce(list.__add__, train_image_clip + train_bbox_clip)
        self.eval_margin = min(eval_all), max(eval_all)
        self.train_margin = min(train_all), max(train_all)

    def get_train_clips(self, seq_lengths: Tuple[int, ...]) -> List[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]:
        clips = functools.reduce(list.__add__, [
            [
                (
                    seq_id,
                    frame_id,
                    {
                        'image': self.train_image_clip[rand_id],
                        'bbox': self.train_bbox_clip[rand_id],
                    }
                )
                for frame_id, rand_id in zip(
                    range(-self.train_margin[0], seq_len - self.train_margin[1]),
                    random.choices(range(len(self.train_image_clip)),
                                   k=seq_len - self.train_margin[1] + self.train_margin[0]),
                )
            ]
            for seq_id, seq_len in enumerate(seq_lengths)
        ])
        random.shuffle(clips)
        return clips

    @functools.lru_cache(maxsize=2)
    def get_eval_clips(self, seq_lengths: Tuple[int, ...]) -> List[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]:
        return functools.reduce(list.__add__, [
            [
                (
                    seq_id,
                    frame_id,
                    {
                        'image': self.eval_image_clip,
                        'bbox': self.eval_bbox_clip,
                    }
                )
                for frame_id in range(-self.eval_margin[0], seq_len - self.eval_margin[1])
            ]
            for seq_id, seq_len in enumerate(seq_lengths)
        ])

    def sample(
            self, train: bool, seq_lengths: List[int]
    ) -> List[Tuple[int, int, Dict[ComponentLiteral, List[int]]]]:
        if train:
            return self.get_train_clips(tuple(seq_lengths))
        else:
            return self.get_eval_clips(tuple(seq_lengths))
