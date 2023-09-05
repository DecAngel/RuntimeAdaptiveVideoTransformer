import json

from .phase_init import PhaseInitMixin, InternalConfigs, ConfigTypes

from ravt.utils.lightning_logger import ravt_logger, get_start_string, get_end_string


class BaseLauncher(PhaseInitMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'summary':
            ravt_logger.info(
                f'\n'
                f'{get_start_string(f"Configs for {self.__class__.__name__}")}'
                f'{json.dumps(configs, default=lambda x: str(x))}'
                f'{get_end_string(f"Configs for {self.__class__.__name__}")}'
            )
        return configs



