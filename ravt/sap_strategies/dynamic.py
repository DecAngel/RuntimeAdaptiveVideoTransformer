from typing import Callable, Tuple, Dict, Optional, List

import numpy as np
import torch

from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import BatchTDict, BatchNDict


class DynamicSchedulingStrategy(BaseSAPStrategy):
    def infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[BatchTDict]],
            process_fn: Callable[[BatchTDict, Optional[Dict]], Tuple[BatchTDict, Dict]],
            output_fn: Callable[[BatchTDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        current_fid = -1
        buffer = None
        runtime_mean = 0.8
        runtime_count = 0
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
            if runtime_mean >= 1:
                runtime_remainder = start_fid - fid
                if runtime_mean < np.floor(runtime_remainder + runtime_mean):
                    continue

            frame['bbox'] = {
                'clip_id': torch.ones(1, 1, 1, dtype=torch.long, device=frame['image_id'].device).long()
            }
            res, buffer = process_fn(frame, buffer)
            output_fn(res)

            runtime_current = time_fn() - start_fid
            runtime_sum = runtime_mean * runtime_count + runtime_current
            runtime_count += 1
            runtime_mean = runtime_sum / runtime_count
