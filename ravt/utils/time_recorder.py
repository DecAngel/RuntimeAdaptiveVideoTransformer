import functools
import sys
import os
import time
from typing import Optional, TextIO
from collections import defaultdict

import torch.multiprocessing as mp
import numpy as np


class TimeRecorder:
    def __init__(self, description: str = 'TimeRecorder', mode: str = 'sum', file: Optional[TextIO] = sys.stdout):
        self.description = description
        assert mode in ['sum', 'avg']
        self.reduce_fn = np.sum if mode == 'sum' else np.mean
        self.file = file if file is not None else open(os.devnull, 'w')
        self._start()

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record(f'{self.description} exit')
        self.print()

    def __str__(self):
        return f'{self.description} total time: {round(self.last_time - self.start_time, 4)}s\n\t' + \
               '\t'.join([f'{k}: {np.round(self.reduce_fn(v), 4).item()}s\n' for k, v in self.t.items()])

    def _start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.t = defaultdict(list)

    def record(self, tag: Optional[str] = None) -> float:
        """Record and return the time elapsed from last call with `tag`.
        If `tag` is None, the time is not recorded, just returned.
        Recorded time with the same `tag` are summed or averaged according to `mode`.

        :param tag: the string tag
        :return: the duration in seconds
        """
        duration = time.time() - self.last_time

        if tag is not None:
            self.t[tag].append(duration)

        self.last_time = time.time()
        return duration

    def get_res_dict(self):
        return self.t

    def print(self):
        print(self.__str__(), file=self.file)


"""
manager = mp.Manager()
time_measure_dict = manager.dict()


def time_measured(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global manager, time_measure_dict
        start_time = time.time()
        res = fn(*args, **kwargs)
        duration = time.time()-start_time
        if fn.__name__ not in time_measure_dict.keys():
            time_measure_dict[fn.__name__] = manager.list([duration])
        else:
            time_measure_dict[fn.__name__].append(duration)
        # print(f'{fn.__name__}: {duration:.3f}s')
        return res
    return wrapper


def time_summary():
    global time_measure_dict
    for k, v in time_measure_dict.items():
        print(f'{k}:\n\ttotal number: {len(v)}\n\tmean time: {sum(v)/len(v):.3f}s')
"""
