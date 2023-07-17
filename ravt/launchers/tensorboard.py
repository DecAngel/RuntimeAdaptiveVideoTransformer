import socket
from tensorboard import program

from f3fusion.configs import output_train_log_dir, tensorboard_port
from f3fusion.utils import f3fusion_logger as logger


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def run_tensorboard():
    if is_port_in_use(tensorboard_port):
        logger.warning(f'Tensorboard port {tensorboard_port} is occupied, assume it\'s already running!')
    else:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', str(output_train_log_dir.joinpath('lightning_logs')), '--bind_all', '--port', str(tensorboard_port)])
        url = tb.launch()
        logger.info(f'Starting tensorboard on {url}.')
