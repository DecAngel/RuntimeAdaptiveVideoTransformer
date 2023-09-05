import math
from typing import Callable

from .base import BaseStrategy
from f3fusion.utils.time_recorder import TimeRecorder


class DadeSchedulingStrategy(BaseStrategy):
    def infer_sequence(self, frame_continuous_fn: Callable[[], float]):
        with TimeRecorder(description=self.__class__.__name__, mode='avg') as tr:
            current_fid = -1
            last_runtime = 1
            self.system.predict_nums = [1]
            while True:
                fid, frame = self.recv_fn()
                if fid is None:
                    break
                elif fid == current_fid:
                    continue

                current_fid = fid
                start_fid = frame_continuous_fn()
                predict_num = min(4, max(1, math.ceil(last_runtime)))

                self.system.predict_nums = [predict_num]
                tr.record('recv')
                res = self.system.infer(frame, fid == 0)
                tr.record('infer')
                self.send_fn(res[-1].array)
                last_runtime = frame_continuous_fn() - start_fid
                tr.record('send')
