from typing import List, get_args

from ..structures import InternalConfigs, ConfigTypes, EnvironmentConfigs


class PhaseInitMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_phase_done: bool = False
        self._init_phase_children: List[PhaseInitMixin] = []

    def __setattr__(self, key, value):
        if isinstance(value, PhaseInitMixin):
            super().__getattribute__('_init_phase_children').append(value)
        return super().__setattr__(key, value)

    def _phase_init_sequence(self) -> List['PhaseInitMixin']:
        if self._init_phase_done:
            return []
        else:
            self._init_phase_done = True

        res = [self]
        for c in self._init_phase_children:
            res.extend(c._phase_init_sequence())
        return res

    def _phase_init_super(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        for s in [super(c, self) for c in self.__class__.mro()[-2::-1]] + [self]:
            if hasattr(s, 'phase_init_impl'):
                try:
                    configs = s.phase_init_impl(phase, configs)
                except NotImplementedError:
                    pass
        return configs

    def add_phase_module(self, module: 'PhaseInitMixin'):
        self._init_phase_children.append(module)

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        raise NotImplementedError()

    def phase_init(self, envs: EnvironmentConfigs) -> InternalConfigs:
        configs = {
            'environment': envs,
            'launcher': {},
            'dataset': {},
            'model': {},
            'evaluation': {},
            'visualization': {},
            'summary': {}
        }
        init_list = self._phase_init_sequence()
        for p in get_args(ConfigTypes):
            for i in init_list:
                configs = i._phase_init_super(p, configs)
        return configs
