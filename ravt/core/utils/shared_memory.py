import functools
import pickle
import ctypes
import re
import subprocess
import multiprocessing.shared_memory as sm
from multiprocessing import resource_tracker
from typing import Tuple, Dict, List, Callable

import numpy as np
from websockets.sync.server import ServerConnection, serve
from websockets.sync.client import connect

from .lightning_logger import ravt_logger as logger
from ..configs import shared_memory_port


class CallOnExit:
    def __init__(self, *callables: Callable):
        self.callables = callables

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for c in self.callables:
            c()


class SharedMemoryServer:
    def __init__(self):
        super().__init__()
        self.close_list: List[sm.SharedMemory] = []
        self.array: Dict[str, int] = {}

        self.safe_margin = 1*(2**30)

    @functools.cached_property
    def available_shared_memory(self) -> int:
        # only care about linux now
        proc = subprocess.Popen(
            'df',
            stdout=subprocess.PIPE, shell=True, text=True,
        )
        x = proc.communicate()[0]
        return int(re.findall(r'(\d*)\s*\d*%\s*/dev/shm', x)[0])*(2**10) - self.safe_margin

    def request_shared_memory(self, name: str, size: int) -> bool:
        if name in self.array:
            if size == self.array[name]:
                logger.info(f'"{name}" already allocated')
                return True
            else:
                logger.error(f'"{name}" allocated but with different ctype or shape!')
                return False
        else:
            if self.available_shared_memory - size > 0:
                self.available_shared_memory -= size
                logger.info(f'allocating "{name}" with total size {size/(2**30):.2f} GB, '
                            f'{self.available_shared_memory/(2**30):.2f} GB available shm left.')
                self.close_list.append(sm.SharedMemory(name, create=True, size=size))
                self.array[name] = size
                return True
            else:
                logger.error(f'"{name}" requests {size/(2**30):.2f} GB, '
                             f'but only {self.available_shared_memory/(2**30):.2f} GB available shm left!')
                return False

    def shutdown(self):
        for s in self.close_list:
            s.close()
            s.unlink()
        self.close_list.clear()

    def handler(self, websocket: ServerConnection):
        for msg in websocket:
            name, size = pickle.loads(msg)
            websocket.send(pickle.dumps(self.request_shared_memory(name, size)))

    def run(self):
        with CallOnExit(self.shutdown):
            with serve(self.handler, '127.0.0.1', shared_memory_port) as server:
                logger.info(f'starting shm server on ws://127.0.0.1:{shared_memory_port}')
                server.serve_forever()


class SharedMemoryClient:
    def __init__(self):
        super().__init__()
        self.close_list: List[sm.SharedMemory] = []

    def test_connection(self) -> bool:
        try:
            c = connect(f'ws://127.0.0.1:{shared_memory_port}', open_timeout=0.5, close_timeout=0.5)
            c.close()
            return True
        except (TimeoutError, ConnectionRefusedError):
            return False

    def request_shared_memory(self, name: str, dtype: object, shape_tuple: Tuple[int, ...]) -> np.ndarray:
        ctype = np.ctypeslib.as_ctypes_type(dtype)
        size = ctypes.sizeof(ctype) * functools.reduce(int.__mul__, shape_tuple)
        with connect(f'ws://127.0.0.1:{shared_memory_port}') as client:
            logger.info(f'allocating "{name}" with total size {size/(2**30):.2f} GB')
            client.send(pickle.dumps((name, size)))
            res = pickle.loads(client.recv())
            if not res:
                raise RuntimeError('Cannot allocate shm memory, please refer to server logs')
        arr = sm.SharedMemory(name, create=False)
        resource_tracker.unregister(arr._name, 'shared_memory')
        self.close_list.append(arr)
        return np.ndarray(shape=shape_tuple, dtype=dtype, buffer=arr.buf)

    def shutdown(self):
        for s in self.close_list:
            s.close()

    def __del__(self):
        self.shutdown()
