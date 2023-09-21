from typing import Optional

from tensorboard import program

from ravt.base_configs import environment_configs
from ravt.core.base_classes import BaseLauncher, launcher_entry
from ravt.core.constants import AllConfigs, PhaseTypes
from ravt.core.utils.lightning_logger import ravt_logger as logger
from ravt.core.utils.ip_address import get_all_local_ip, is_port_in_use


class TensorboardLauncher(BaseLauncher):
    def __init__(self, envs: Optional[AllConfigs] = None):
        super().__init__(envs or environment_configs)
        self.port = None
        self.output_train_log_dir = None

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'launcher':
            self.port = configs['environment']['tensorboard_port']
            self.output_train_log_dir = configs['environment']['output_train_log_dir']
        return configs

    @launcher_entry()
    def run(self):
        if is_port_in_use(self.port):
            logger.warning(f'Tensorboard port {self.port} is occupied, assume it\'s already running!')
        else:
            tb = program.TensorBoard()
            tb.configure(
                argv=[None, '--logdir', str(self.output_train_log_dir), '--bind_all', '--port', str(self.port)]
            )
            logger.info(
                f'Starting tensorboard on '
                f'{", ".join(f"http://{i}:{self.port}" for i in get_all_local_ip())}'
            )
            tb.main()
