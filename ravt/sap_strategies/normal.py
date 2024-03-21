from typing import Callable, Tuple, Dict, Optional, List

import torch

from ravt.core.base_classes import BaseSAPStrategy
from ravt.core.constants import BatchNDict, BatchTDict


class NormalStrategy(BaseSAPStrategy):
    def infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[BatchTDict]],
            process_fn: Callable[[BatchTDict, Optional[Dict]], Tuple[BatchTDict, Dict]],
            output_fn: Callable[[BatchTDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        current_fid = -1
        buffer = None
        while True:
            frame = input_fn()
            if frame is None:
                # end of sequence
                break

            fid = frame['frame_id'].squeeze().cpu().numpy().item()
            if fid == current_fid:
                continue
            else:
                current_fid = fid

            frame['bbox'] = {
                'clip_id': torch.ones(1, 1, 1, dtype=torch.long, device=frame['image_id'].device).long()
            }

            res, buffer = process_fn(frame, buffer)
            output_fn(res)
