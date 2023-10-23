from typing import Callable, Tuple, Dict, Optional, List

import numpy as np

from ravt.core.base_classes import BaseSAPStrategy


class NormalStrategy(BaseSAPStrategy):
    def infer_sequence(
            self,
            input_fn: Callable[[], Tuple[Optional[int], np.ndarray]],
            process_fn: Callable[[np.ndarray, Optional[Dict], List[int], List[int]], Tuple[np.ndarray, Dict]],
            output_fn: Callable[[np.ndarray], None],
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
            output_fn(res)
