from typing import Callable, Tuple, Dict, Optional, List

from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType, BBoxInferenceType


class NormalStrategy(BaseSAPStrategy):
    def infer_sequence_impl(
            self,
            input_fn: Callable[[], Tuple[Optional[int], ImageInferenceType]],
            process_fn: Callable[
                [ImageInferenceType, Optional[Dict], Optional[List[int]], Optional[List[int]]],
                Tuple[BBoxesInferenceType, Dict]
            ],
            output_fn: Callable[[BBoxInferenceType], None],
            time_fn: Callable[[], float],
    ):
        current_fid = -1
        buffer = None
        while True:
            fid, frame = input_fn()
            if fid is None:
                # end of sequence
                break
            elif fid == current_fid:
                continue
            else:
                current_fid = fid

            res, buffer = process_fn(frame, buffer, [-1], [1])
            output_fn(res[0])
