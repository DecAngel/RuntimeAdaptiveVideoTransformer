from typing import Callable, Tuple, Dict, Optional, List

import numpy as np

from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType, BBoxInferenceType


class DynamicSchedulingStrategy(BaseSAPStrategy):
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
        runtime_mean = 0.8
        runtime_count = 0
        while True:
            fid, frame = input_fn()
            if fid is None:
                break
            elif fid == current_fid:
                continue
            else:
                current_fid = fid

            start_fid = time_fn()
            if runtime_mean >= 1:
                runtime_remainder = start_fid - fid
                if runtime_mean < np.floor(runtime_remainder + runtime_mean):
                    continue

            res, buffer = process_fn(frame, buffer, None, None)
            output_fn(res[0])

            runtime_current = time_fn() - start_fid
            runtime_sum = runtime_mean * runtime_count + runtime_current
            runtime_count += 1
            runtime_mean = runtime_sum / runtime_count
