import functools
import json
from typing import Optional

from ..constants import AllConfigs, all_configs_empty
from ..utils.phase_init import PhaseInitMixin
from ..utils.lightning_logger import ravt_logger as logger, get_start_string, get_end_string


def launcher_entry(additional_configs: Optional[AllConfigs] = None):
    def wrapper_fn(launcher_fn):
        @functools.wraps(launcher_fn)
        def inner(self: BaseLauncher, *args, **kwargs):
            envs = self.envs
            if additional_configs is not None:
                for k, v in additional_configs.items():
                    envs[k] |= v
            configs = self.phase_init(envs)
            logger.info(
                f'{get_start_string(f"Configs for {self.__class__.__name__}")}\n'
                f'{json.dumps(configs, indent=2, default=lambda x: str(x))}\n'
                f'{get_end_string(f"Configs for {self.__class__.__name__}")}\n'
            )
            return launcher_fn(self, *args, **kwargs)
        return inner
    return wrapper_fn


class BaseLauncher(PhaseInitMixin):
    def __init__(self, envs: AllConfigs):
        super().__init__()
        self.envs = all_configs_empty
        self.envs.update(**envs)
