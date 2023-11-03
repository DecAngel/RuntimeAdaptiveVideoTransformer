from typing import Callable, Tuple, Dict, List, Optional

import numpy as np


class BaseSAPStrategy:
    def __init__(self):
        super().__init__()

    def infer_sequence(
            self,
            input_fn: Callable[[], Tuple[Optional[int], np.ndarray]],
            process_fn: Callable[[np.ndarray, Optional[Dict]], Tuple[np.ndarray, Dict]],
            output_fn: Callable[[np.ndarray], None],
            time_fn: Callable[[], float],
    ):
        raise NotImplementedError()
