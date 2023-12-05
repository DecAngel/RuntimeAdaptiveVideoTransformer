import contextlib
import functools
import io
import json
import signal
import sys
import os
import platform
import socket
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from sap_toolkit.client import EvalClient
from sap_toolkit.generated import eval_server_pb2
from tqdm import tqdm

from ..base_classes import BaseSystem
from ..constants import BatchTDict, BatchNDict
from ..utils.lightning_logger import ravt_logger as logger
from ..utils.array_operations import remove_pad_along
from ..configs import output_sap_log_dir


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

        output_dict = {}
        while True:
            output = self.proc.stdout.readline()
            if '=' in output:
                o = output.rsplit('=')
                try:
                    output_dict['='.join(o[:-1])] = float(o[-1])
                except ValueError:
                    pass
            if 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' in output:
                break
        return output_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.communicate('close\n')
        os.remove(self.config_path)
        self.proc = None
        self.config_path = None


class SAPClient(EvalClient):
    def generate(self, results_file='results.json'):
        self.result_stub.GenResults(eval_server_pb2.String(value=results_file))

    def close(self):
        self.result_channel.close()
        self.existing_shm.close()
        self.results_shm.close()
        print('Shutdown', flush=True)


# TODO: sap not completed
def run_sap(
        system: BaseSystem,
        sap_factor: float = 1.0,
        dataset_resize_ratio: float = 2,
        dataset_fps: int = 30,
        device_id: Optional[int] = None,
        debug: bool = __debug__,
) -> Dict[str, Any]:
    if platform.system() != 'Linux':
        raise EnvironmentError('sAP evaluation is only supported on Linux!')

    system = system.eval().to(torch.device(f'cuda:{device_id or 0}'))

    with SAPServer(
            data_dir=system.data_sources['test'].img_dir,
            ann_file=system.data_sources['test'].ann_file,
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
        signal.signal(signal.SIGINT, client.close)
        signal.signal(signal.SIGTERM, client.close)

        try:
            # load coco dataset
            with contextlib.redirect_stdout(io.StringIO()):
                coco = COCO(str(system.data_sources['test'].ann_file()))
            seqs = coco.dataset['sequences']

            def resize_image(img: np.ndarray) -> np.ndarray:
                h, w, c = img.shape
                return cv2.resize(
                    img, (w // dataset_resize_ratio, h // dataset_resize_ratio)
                )

            def recv_fn() -> Optional[BatchNDict]:
                frame_id_next, frame = client.get_frame()
                if frame_id_next is not None:
                    original_size = np.array(frame.shape[:2])
                    frame = resize_image(frame)[..., [2, 1, 0]].transpose(2, 0, 1)
                    return {
                        'image_id': np.array(frame_id_next),
                        'seq_id': np.array(0),
                        'frame_id': np.array(frame_id_next),
                        'image': {
                            'image': frame,
                            'original_size': original_size,
                            'clip_id': np.array(0),
                        }
                    }
                else:
                    return None

            def send_fn(batch: BatchNDict) -> None:
                coordinate = remove_pad_along(batch['bbox']['coordinate'], axis=0) * dataset_resize_ratio
                probability = batch['bbox']['probability'][:coordinate.shape[0]]
                label = batch['bbox']['label'][:coordinate.shape[0]]
                client.send_result_to_server(coordinate, probability, label)
                return None

            def time_fn(start_time: float):
                return (time.perf_counter() - start_time) * dataset_fps * sap_factor

            # warm_up
            for _ in range(2):
                system.inference(system.example_input_array[0], None)
                # system.inference(resize_image(np.zeros((1200, 1920, 3), dtype=np.uint8)), None)
            torch.cuda.synchronize(device=torch.device(f'cuda:{device_id or 0}'))

            for i, seq_id in enumerate(tqdm(seqs)):
                client.request_stream(seq_id)
                t_start = client.get_stream_start_time()
                system.strategy.infer_sequence(
                    input_fn=recv_fn,
                    process_fn=system.inference,
                    output_fn=send_fn,
                    time_fn=functools.partial(time_fn, t_start)
                )
                client.stop_stream()
                if i == 0:
                    system.strategy.plot_process_time()
            filename = f'{int(time.time()) % 1000000000}.json'
            client.generate(filename)
            return server.get_result(filename)

        except KeyboardInterrupt as e:
            logger.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
            raise
        finally:
            client.close()
