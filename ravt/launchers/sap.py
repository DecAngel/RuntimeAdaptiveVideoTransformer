import contextlib
import functools
import io
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
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pycocotools.coco import COCO
from sap_toolkit.client import EvalClient
from sap_toolkit.generated import eval_server_pb2
from tqdm import tqdm

from ravt.core.base_classes import BaseSystem, BaseDataSource, BaseSAPStrategy
from ravt.core.constants import ImageInferenceType, BBoxInferenceType
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.array_operations import remove_pad_along
from ravt.core.utils.time_recorder import TimeRecorder
from ravt.core.configs import output_sap_log_dir


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

    def get_result(self, results_file: str = 'results.json') -> Dict[str, Any]:
        self.proc.stdin.write(f'evaluate {results_file}\n')
        self.proc.stdin.flush()

        """
        keyword = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
        while True:
            result = self.proc.stdout.readline()
            if keyword in result:
                index = result.find(keyword)
                return float(result[index+len(keyword):index+len(keyword)+5])
        """
        output_dict = {}
        while self.proc.stdout.readable():
            output = self.proc.stdout.readline()
            print(output)
            if '=' in output:
                v, k = output.rsplit('=', 2)
                try:
                    output_dict[k] = float(v)
                except ValueError:
                    pass
        return output_dict

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


def run_sap(
        system: BaseSystem,
        sap_factor: float = 1.0,
        dataset_resize_ratio: float = 2,
        dataset_fps: int = 30,
        device_id: Optional[int] = None,
        debug: bool = __debug__,
        seed: Optional[int] = None,
) -> Dict[str, Any]:
    if seed is None:
        raise ValueError('seed must be set at the beginning of the exp!')

    if platform.system() != 'Linux':
        raise EnvironmentError('sAP evaluation is only supported on Linux!')

    system = system.eval().to(torch.device(f'cuda:{device_id or 0}'))

    with SAPServer(
            data_dir=system.data_source.get_image_dir('test'),
            ann_file=system.data_source.get_ann_file('test'),
            output_dir=output_sap_log_dir,
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
            with contextlib.redirect_stdout(io.StringIO()):
                coco = COCO(str(system.data_source.get_ann_file('test')))
            seqs = coco.dataset['sequences']
            tr = TimeRecorder('SAP Profile', mode='avg')

            def resize_input(img: ImageInferenceType) -> ImageInferenceType:
                h, w, c = img.shape
                return cv2.resize(
                    img, (w // dataset_resize_ratio, h // dataset_resize_ratio), interpolation=cv2.INTER_NEAREST
                )

            def recv_fn():
                frame_id_next, frame = client.get_frame()
                frame = resize_input(frame) if frame_id_next is not None else None
                return frame_id_next, frame

            def send_fn(array: BBoxInferenceType):
                array = remove_pad_along(array, axis=0)
                client.send_result_to_server(array[:, :4] * dataset_resize_ratio, array[:, 4], array[:, 5])
                tr.record('output_fn')
                return None

            def time_fn(start_time: float):
                return (time.perf_counter() - start_time) * dataset_fps

            # warm_up
            system.inference(resize_input(np.zeros((1200, 1920, 3), dtype=np.int8)), None)

            for seq_id in tqdm(seqs):
                client.request_stream(seq_id)
                t_start = client.get_stream_start_time()
                tr.record()
                system.strategy.infer_sequence(
                    input_fn=recv_fn,
                    process_fn=system.inference,
                    output_fn=send_fn,
                    time_fn=functools.partial(time_fn, t_start)
                )
                client.stop_stream()

            tr.print()
            filename = f'{seed}_{int(time.time()) % 10000}.json'
            client.generate(filename)
            return server.get_result(filename)

        except KeyboardInterrupt as e:
            logger.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
            raise
        finally:
            client.close()
