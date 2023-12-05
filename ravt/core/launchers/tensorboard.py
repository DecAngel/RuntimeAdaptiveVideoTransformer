from tensorboard import program

from ..utils.lightning_logger import ravt_logger as logger
from ..utils.ip_address import get_all_local_ip, is_port_in_use
from ..configs import tensorboard_port, output_train_log_dir


def run_tb_server():
    if is_port_in_use(tensorboard_port):
        logger.warning(f'Tensorboard port {tensorboard_port} is occupied, assume it\'s already running!')
    else:
        tb = program.TensorBoard()
        tb.configure(
            argv=[None, '--logdir', str(output_train_log_dir), '--bind_all', '--port', str(tensorboard_port)]
        )
        logger.info(
            f'Starting tensorboard on '
            f'{", ".join(f"http://{i}:{tensorboard_port}" for i in get_all_local_ip())}'
        )
        tb.main()
