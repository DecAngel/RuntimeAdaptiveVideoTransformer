import functools
import pickle
import ctypes
import multiprocessing.shared_memory as sm
from multiprocessing import resource_tracker
from typing import Tuple, Any, Dict, List, Optional

import numpy as np
from websockets.sync.server import ServerConnection, serve
from websockets.sync.client import ClientConnection, connect

from ravt.core.constants import PhaseTypes, AllConfigs
from ravt.core.utils.contexts import CallOnExit
from ravt.core.utils.phase_init import PhaseInitMixin
from ravt.core.utils.lightning_logger import ravt_logger as logger


class SharedMemoryServer(PhaseInitMixin):
    def __init__(self):
        super().__init__()
        self.close_list: List[sm.SharedMemory] = []
        self.array: Dict[str, int] = {}
        self.port: Optional[int] = None

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'dataset':
            self.port = configs['environment']['shared_memory_port']
        return configs

    def request_shared_memory(self, name: str, size: int) -> bool:
        if name in self.array:
            if size == self.array[name]:
                logger.info(f'"{name}" already allocated')
                return True
            else:
                logger.error(f'"{name}" allocated but with different ctype or shape!')
                return False
        else:
            logger.info(f'allocating "{name}" with total size {size/(2**30):.2f} GB')
            self.close_list.append(sm.SharedMemory(name, create=True, size=size))
            self.array[name] = size
            logger.info(f'"{name}" allocated')
            return True

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
            with serve(self.handler, '127.0.0.1', self.port) as server:
                server.serve_forever()


class SharedMemoryClient(PhaseInitMixin):
    def __init__(self):
        super().__init__()
        self.close_list: List[sm.SharedMemory] = []
        self.port: Optional[int] = None

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'dataset':
            self.port = configs['environment']['shared_memory_port']
        return configs

    def test_connection(self) -> bool:
        try:
            c = connect(f'ws://127.0.0.1:{self.port}', open_timeout=1, close_timeout=1)
            c.close()
            return True
        except TimeoutError:
            return False

    def request_shared_memory(self, name: str, dtype: object, shape_tuple: Tuple[int, ...]) -> np.ndarray:
        ctype = np.ctypeslib.as_ctypes_type(dtype)
        size = ctypes.sizeof(ctype) * functools.reduce(int.__mul__, shape_tuple)
        with connect(f'ws://127.0.0.1:{self.port}') as client:
            logger.info(f'allocating "{name}" with total size {size/(2**30):.2f} GB')
            client.send(pickle.dumps((name, size)))
            res = pickle.loads(client.recv())
            if res:
                logger.info(f'"{name}" allocated')
            else:
                raise ValueError(f'{name} already allocated with a different size!')
        arr = sm.SharedMemory(name, create=False)
        resource_tracker.unregister(arr._name, 'shared_memory')
        self.close_list.append(arr)
        return np.ndarray(shape=shape_tuple, dtype=dtype, buffer=arr.buf)

    def shutdown(self):
        for s in self.close_list:
            s.close()

    def __del__(self):
        self.shutdown()
