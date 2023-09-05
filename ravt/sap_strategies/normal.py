from typing import Callable

from .base import BaseStrategy
from f3fusion.utils.time_recorder import TimeRecorder


class NormalStrategy(BaseStrategy):
    def infer_sequence(self, frame_continuous_fn: Callable[[], float]):
        with TimeRecorder(description=self.__class__.__name__, mode='avg') as tr:
            current_fid = -1
            self.system.predict_nums = [1]
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
