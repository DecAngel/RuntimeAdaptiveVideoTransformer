from typing import Callable, Tuple

import numpy as np

from ..models import BaseSystem


class BaseStrategy:
    def __init__(
            self,
            system: BaseSystem,
            recv_fn: Callable[[], Tuple[int, np.ndarray]],
            send_fn: Callable[[np.ndarray], None],
    ):
        self.system = system
        self.recv_fn = recv_fn
        self.send_fn = send_fn

    def infer_sequence(self, frame_continuous_fn: Callable[[], float]):
        raise NotImplementedError()
