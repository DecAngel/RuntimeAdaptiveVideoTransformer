from typing import Callable, Tuple, Dict, List, Optional

from ..constants import ImageInferenceType, BBoxesInferenceType, BBoxInferenceType


class BaseSAPStrategy:
    def __init__(self):
        super().__init__()

    def infer_sequence(
            self,
            input_fn: Callable[[], Tuple[Optional[int], ImageInferenceType]],
            process_fn: Callable[
                [ImageInferenceType, Optional[Dict], Optional[List[int]], Optional[List[int]]],
                Tuple[BBoxesInferenceType, Dict]
            ],
            output_fn: Callable[[BBoxInferenceType], None],
            time_fn: Callable[[], float],
    ):
        raise NotImplementedError()
