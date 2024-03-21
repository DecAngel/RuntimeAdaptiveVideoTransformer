import math
from typing import Callable, Tuple, Dict, Optional, List

import numpy as np
import torch

from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import BatchTDict


class MSCASchedulingStrategy(BaseSAPStrategy):
    def infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[BatchTDict]],
            process_fn: Callable[[BatchTDict, Optional[Dict]], Tuple[BatchTDict, Dict]],
            output_fn: Callable[[BatchTDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        current_fid = -1
        buffer = None
        runtimes = [1]
        runtime_window_size = 5

        while True:
            frame = input_fn()
            if frame is None:
                break

            fid = frame['frame_id'].squeeze().cpu().numpy().item()
            if fid == current_fid:
                continue
            else:
                current_fid = fid

            start_fid = time_fn()
            predict_num = min(10, max(1, math.ceil(sum(runtimes) / len(runtimes))))

            frame['image']['clip_id'] = fid
            frame['bbox'] = {
                'clip_id': torch.ones(1, 1, 1, dtype=torch.long, device=frame['image_id'].device).long() * predict_num
            }

            res, buffer = process_fn(frame, buffer)
            output_fn(res[0])

            last_runtime = time_fn() - start_fid
            runtimes.append(last_runtime)
            if len(runtimes) > runtime_window_size:
                runtimes.pop(0)
