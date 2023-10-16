import functools
import json
import os
import platform
import random
import socket
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Type, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from sap_toolkit.client import EvalClient
from sap_toolkit.generated import eval_server_pb2
from tqdm import tqdm

from ..sap_strategies import BaseStrategy
from ..models import BaseSystem
from f3fusion.configs import dataset_argoverse_dir, output_sap_log_dir
from ravt.core.constants import AllConfigs, PhaseTypes
from ravt.core.base_classes import BaseLauncher, BaseSystem, BaseDataSource, launcher_entry
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.array_operations import remove_pad_along

from ..base_configs import environment_configs


class SAPServer:
    def __init__(self, data_dir: Path, ann_file: Path, output_dir: Path, sap_factor: float = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.output_dir = output_dir
        self.sap_factor = sap_factor
        self.config_path: Optional[Path] = None
        self.proc: Optional[subprocess.Popen] = None

    @staticmethod
    def find_2_unused_ports() -> Tuple[int, int]:
        # create a temporary socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                # bind the sockets to random addresses
                s1.bind(('', 0))
                s2.bind(('', 0))
                # retrieve the port numbers that was allocated
                port1 = s1.getsockname()[1]
                port2 = s2.getsockname()[1]
        return port1, port2

    def __enter__(self):
        # create temporary configs
        p1, p2 = self.find_2_unused_ports()
        config = {
            "image_service_port": p1,
            "result_service_port": p2,
            "loopback_ip": "127.0.0.1"
        }
        self.config_path = self.output_dir.joinpath(f'{p1}_{p2}.json')
        self.config_path.write_text(json.dumps(config))

        # start server
        self.proc = subprocess.Popen(
            ' '.join([
                'python', '-m', 'sap_toolkit.server',
                '--data-root', f'{str(self.data_dir.resolve())}',
                '--annot-path', f'{str(self.ann_file.resolve())}',
                '--overwrite',
                '--eval-config', f'{str(self.config_path.resolve())}',
                '--out-dir', f'{str(self.output_dir.resolve())}',
                '--perf-factor', str(self.sap_factor),
            ]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(self.output_dir.resolve()),
            shell=True,
            text=True,
        )

        keyword = 'Welcome'
        while True:
            result = self.proc.stdout.readline()
            if keyword in result:
                break

        return self

    def get_result(self, results_file: str = 'results.json') -> float:
        self.proc.stdin.write(f'evaluate {results_file}\n')
        self.proc.stdin.flush()

        keyword = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
        while True:
            result = self.proc.stdout.readline()
            if keyword in result:
                index = result.find(keyword)
                return float(result[index+len(keyword):index+len(keyword)+5])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.communicate('close\n')
        os.remove(self.config_path)
        self.proc = None
        self.config_path = None


class SAPClient(EvalClient):
    def generate(self, results_file='results.json'):
        self.result_stub.GenResults(eval_server_pb2.String(value=results_file))

    def close(self, results_file='results.json'):
        self.result_channel.close()
        self.existing_shm.close()
        self.results_shm.close()


class SAPLauncher(BaseLauncher):


def run_sap(
        system: BaseSystem,
        sap_strategy: Type[BaseStrategy],
        sap_factor: float = 1.0,
        dataset_resize_ratio: int = 2,
        dataset_fps: int = 30,
        device: Optional[int] = None
) -> float:
    if platform.system() != 'Linux':
        raise EnvironmentError('sAP evaluation is only supported on Linux!')

    device = device or 0
    system.eval()
    system = system.to(torch.device(f'cuda:{device}'))

    def dataset_resizer(img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        return cv2.resize(img, (w // dataset_resize_ratio, h // dataset_resize_ratio), interpolation=cv2.INTER_NEAREST)

    with SAPServer(sap_factor=sap_factor) as server:
        # start client
        client_state = (
            mp.Value('i', -1, lock=True),
            mp.Event(),
            mp.Manager().dict(),
        )
        client = SAPClient(json.loads(server.config_path.read_text()), client_state, verbose=True)
        client.stream_start_time = mp.Value('d', 0.0, lock=True)

        try:
            # load coco dataset
            coco = COCO(str(dataset_argoverse_dir.joinpath("Argoverse-HD", "annotations", "val.json")))
            seqs = coco.dataset['sequences']

            # warm_up
            system.infer(dataset_resizer(np.zeros((1200, 1920, 3), dtype=np.int8)))

            def recv_fn():
                frame_id_next, frame = client.get_frame()
                return frame_id_next, dataset_resizer(frame) if frame_id_next is not None else None

            def send_fn(array: np.ndarray):
                array = remove_pad_along(array, axis=0)
                client.send_result_to_server(array[:, :4] * dataset_resize_ratio, array[:, 4], array[:, 5])
                return None

            def time_fn(start_time: float):
                return (time.perf_counter() - start_time) * dataset_fps

            strategy = sap_strategy(system, send_fn=send_fn, recv_fn=recv_fn)

            for seq_id in tqdm(seqs):
                client.request_stream(seq_id)
                t_start = client.get_stream_start_time()
                strategy.infer_sequence(functools.partial(time_fn, t_start))
                client.stop_stream()

            filename = f'{int(random.randrange(0, 10000000)+time.time()) % 10000000}.json'
            client.generate(filename)
            return server.get_result(filename)

        except KeyboardInterrupt as e:
            logger.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
            raise
        finally:
            client.close()
