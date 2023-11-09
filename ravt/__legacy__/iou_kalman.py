from typing import Callable, Tuple

import numpy as np

from .base import BaseStrategy, BaseSystem
from ravt.__legacy__.f3fusion import TimeRecorder


class IOUKalmanStrategy(BaseStrategy):
    def __init__(
            self,
            system: BaseSystem,
            recv_fn: Callable[[], Tuple[int, np.ndarray]],
            send_fn: Callable[[np.ndarray], None],
    ):
        super().__init__(system, recv_fn, send_fn)
        self.X_init = np.zeros((1, 8), dtype=np.float)
        self.P_init = 3*np.ones((1, 8), dtype=np.float)
        self.A = np.eye(8) + np.eye(8, k=4)
        self.C = np.eye(4, 8)
        self.Q = np.ones((1, 8), dtype=np.float)
        self.R = np.ones()
        # TODO: complete

    def kf_init(self):
        pass

    def kf_update(self):
        pass

    def kf_predict(self):
        pass

    def iou_match(self):
        pass

    def infer_sequence(self, frame_continuous_fn: Callable[[], float]):
        with TimeRecorder(description=self.__class__.__name__, mode='avg') as tr:
            current_fid = -1
            while True:
                fid, frame = self.recv_fn()
                if fid is None:
                    break
                elif fid == current_fid:
                    continue
                else:
                    current_fid = fid
                tr.record('recv')
                res = self.system.infer(frame, fid == 0)
                tr.record('infer')
                self.send_fn(res[-1].array)
                tr.record('send')
