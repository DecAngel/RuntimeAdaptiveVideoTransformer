import math
from typing import Callable, Tuple, Dict, Optional, List

import numpy as np
from ravt.core.base_classes import BaseSAPStrategy


class DadeSchedulingStrategy(BaseSAPStrategy):
    def infer_sequence(
            self,
            input_fn: Callable[[], Tuple[Optional[int], np.ndarray]],
            process_fn: Callable[[np.ndarray, Optional[Dict], List[int], List[int]], Tuple[np.ndarray, Dict]],
            output_fn: Callable[[np.ndarray], None],
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

            res, buffer = process_fn(frame, buffer, [-3, -2, -1], [predict_num])
            output_fn(res)

            last_runtime = time_fn() - start_fid
