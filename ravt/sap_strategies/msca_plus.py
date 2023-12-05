import math
from typing import Callable, Tuple, Dict, Optional, List

import numpy as np
from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxesInferenceType, BBoxInferenceType


class MSCAPlusSchedulingStrategy(BaseSAPStrategy):
    def __init__(self):
        super().__init__()
        self.time_precision = 0.001
        self.predicted_frame_id = 0
        self.predicted_results: List[BBoxInferenceType] = []
        self.worker_switch = False

    def output_worker(self, output_fn: Callable[[BBoxInferenceType], None], time_fn: Callable[[], float]):
        while self.worker_switch:
            cur_time = time_fn()
            if cur_time < self.predicted_frame_id - self.time_precision * 2:
                # not yet
                pass
            elif len(self.predicted_results) > 0:
                # output
                pass

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
        runtimes = [1]
        runtime_window_size = 5
        while True:
            fid, frame = input_fn()
            if fid is None:
                break
            elif fid == current_fid:
                continue
            else:
                current_fid = fid

            start_fid = time_fn()
            predict_num = min(10, max(1, math.ceil(sum(runtimes) / len(runtimes))))

            res, buffer = process_fn(frame, buffer, list(range(-predict_num, 0)), [predict_num])
            output_fn(res[0])

            last_runtime = time_fn() - start_fid
            runtimes.append(last_runtime)
            if len(runtimes) > runtime_window_size:
                runtimes.pop(0)
