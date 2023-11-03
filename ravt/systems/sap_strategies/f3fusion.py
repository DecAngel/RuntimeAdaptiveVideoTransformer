import functools
import math
import threading
import time
from typing import Callable, Tuple, List

import numpy as np

from ..data_structures import NDArrayBoundingBox
from ..models import F3FusionSystem
from .base import BaseStrategy
from f3fusion.utils.time_recorder import TimeRecorder


class F3FusionStrategy(BaseStrategy):
    def __init__(
            self,
            system: F3FusionSystem,
            recv_fn: Callable[[], Tuple[int, np.ndarray]],
            send_fn: Callable[[np.ndarray], None],
    ):
        super().__init__(system, recv_fn, send_fn)
        self.buffer: List[Tuple[float, NDArrayBoundingBox]] = []
        self.buffer_lock = threading.Lock()
        self.flag = threading.Event()
        # TODO: reconsider

    def timestamped_output(self, frame_continuous_fn: Callable[[], float]):
        while not self.flag.is_set():
            with self.buffer_lock:
                cur_time = frame_continuous_fn()
                while len(self.buffer) != 0 and self.buffer[0][0] < cur_time:
                    self.buffer.pop(0)
                if len(self.buffer) != 0:
                    self.send_fn(self.buffer.pop(0)[1].array)

            time.sleep(0.001)

    def infer_sequence(self, frame_continuous_fn: Callable[[], float]):
        with TimeRecorder(description=self.__class__.__name__, mode='avg') as tr:
            current_fid = -1
            runtime_mean = 1
            t = threading.Thread(target=functools.partial(self.timestamped_output, frame_continuous_fn), daemon=True)
            t.start()
            self.system.predict_nums = [1]
            while True:
                fid, frame = self.recv_fn()
                if fid is None:
                    break
                elif fid == current_fid:
                    continue
                else:
                    current_fid = fid

                start_fid = frame_continuous_fn()
                est_start = int(math.ceil(start_fid + runtime_mean))
                est_end = max(int(math.ceil(start_fid + runtime_mean*2)), est_start + 1)
                est_predict_range = list(range(est_start, est_end))
                self.system.model.predict_nums = est_predict_range
                tr.record('recv')
                res = self.system.infer(frame, fid == 0)
                tr.record('infer')
                with self.buffer_lock:
                    for f, r in zip(est_predict_range, res):
                        self.buffer.append((f, r))

                runtime_current = frame_continuous_fn() - start_fid
                runtime_mean = runtime_mean * 0.5 + runtime_current * 0.5
                tr.record('send')
