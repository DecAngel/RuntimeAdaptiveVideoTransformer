from typing import Callable

import numpy as np

from .base import BaseStrategy, BaseSystem
from f3fusion.utils.time_recorder import TimeRecorder


class DynamicSchedulingStrategy(BaseStrategy):
    def infer_sequence(self, frame_continuous_fn: Callable[[], float]):
        with TimeRecorder(description=self.__class__.__name__, mode='avg') as tr:
            current_fid = -1
            runtime_mean = 0.8
            runtime_count = 0
            self.system.predict_nums = [1]
            while True:
                fid, frame = self.recv_fn()
                if fid is None:
                    break
                elif fid == current_fid:
                    continue

                current_fid = fid
                start_fid = frame_continuous_fn()
                if runtime_mean >= 1:
                    runtime_remainder = start_fid - fid
                    if runtime_mean < np.floor(runtime_remainder + runtime_mean):
                        continue

                tr.record('recv')
                res = self.system.infer(frame, fid == 0)
                tr.record('infer')
                self.send_fn(res[-1].array)
                runtime_current = frame_continuous_fn() - start_fid
                runtime_sum = runtime_mean*runtime_count + runtime_current
                runtime_count += 1
                runtime_mean = runtime_sum / runtime_count
                tr.record('send')
