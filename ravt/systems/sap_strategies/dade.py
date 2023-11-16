import math
from typing import Callable, Tuple, Dict, Optional, List

import numpy as np
from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType, BBoxInferenceType


class DadeSchedulingStrategy(BaseSAPStrategy):
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
        last_runtime = 1
        while True:
            fid, frame = input_fn()
            if fid is None:
                break
            elif fid == current_fid:
                continue
            else:
                current_fid = fid

            start_fid = time_fn()
            predict_num = min(4, max(1, math.ceil(last_runtime)))

            res, buffer = process_fn(frame, buffer, [-predict_num], [predict_num])
            output_fn(res[0])

            last_runtime = time_fn() - start_fid
