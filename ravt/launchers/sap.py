import functools
import json
import sys
import os
import platform
import random
import socket
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pycocotools.coco import COCO
from sap_toolkit.client import EvalClient
from sap_toolkit.generated import eval_server_pb2
from tqdm import tqdm

from ravt.core.constants import AllConfigs, PhaseTypes
from ravt.core.base_classes import BaseLauncher, BaseSystem, BaseDataSource, launcher_entry, BaseSAPStrategy
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.array_operations import remove_pad_along
from ravt.core.utils.time_recorder import TimeRecorder

from ravt.core.configs import environment_configs


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
                sys.executable, '-m', 'sap_toolkit.server',
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
    def __init__(
            self, model: BaseSystem, data_source: BaseDataSource, sap_strategy: BaseSAPStrategy,
            envs: Optional[AllConfigs] = None,
            device_id: Optional[int] = None,
            debug: bool = __debug__,
            seed: Optional[int] = None,
    ):
        super().__init__(envs or environment_configs)
        self.model = model.eval().to(torch.device(f'cuda:{device_id or 0}'))
        self.data_source = data_source
        self.sap_strategy = sap_strategy
        self.device_id = device_id or 0
        self.debug = debug
        self.seed = pl.seed_everything(seed)

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'evaluation':
            self.output_dir = configs['environment']['output_sap_log_dir']
            self.data_dir = configs['extra']['image_dir']
            self.ann_file = configs['extra']['test_coco_file']
            self.original_size = configs['internal']['original_size']
        return configs

    @launcher_entry()
    def test_sap(self, sap_factor: float = 1.0, dataset_resize_ratio: float = 2, dataset_fps: int = 30):
        if platform.system() != 'Linux':
            raise EnvironmentError('sAP evaluation is only supported on Linux!')

        with SAPServer(
            data_dir=self.data_dir,
            ann_file=self.ann_file,
            output_dir=self.output_dir,
            sap_factor=sap_factor
        ) as server:
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
                coco = COCO(str(self.ann_file))
                seqs = coco.dataset['sequences']
                tr = TimeRecorder('SAP Profile', mode='avg')

                def dataset_resizer(img: np.ndarray) -> np.ndarray:
                    h, w, c = img.shape
                    return cv2.resize(
                        img, (w // dataset_resize_ratio, h // dataset_resize_ratio), interpolation=cv2.INTER_NEAREST
                    )

                def recv_fn():
                    frame_id_next, frame = client.get_frame()
                    frame = dataset_resizer(frame) if frame_id_next is not None else None
                    return frame_id_next, frame

                def send_fn(array: np.ndarray):
                    array = remove_pad_along(array, axis=0)
                    client.send_result_to_server(array[:, :4] * dataset_resize_ratio, array[:, 4], array[:, 5])
                    tr.record('output_fn')
                    return None

                def time_fn(start_time: float):
                    return (time.perf_counter() - start_time) * dataset_fps

                # warm_up
                self.model.inference(dataset_resizer(np.zeros((*self.original_size, 3), dtype=np.int8)))

                for seq_id in tqdm(seqs):
                    client.request_stream(seq_id)
                    t_start = client.get_stream_start_time()
                    tr.record()
                    self.sap_strategy.infer_sequence(
                        input_fn=recv_fn,
                        process_fn=self.model.inference,
                        output_fn=send_fn,
                        time_fn=functools.partial(time_fn, t_start)
                    )
                    client.stop_stream()

                tr.print()
                filename = f'{int(random.randrange(0, 10000000)+time.time()) % 10000000}.json'
                client.generate(filename)
                return server.get_result(filename)

            except KeyboardInterrupt as e:
                logger.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
                raise
            finally:
                client.close()
