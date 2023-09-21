from typing import List, get_args

from ravt.core.constants import PhaseTypes, AllConfigs


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

    def _phase_init_super(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        for c in self.__class__.mro()[::-1]:
            if 'phase_init_impl' in c.__dict__:
                configs = getattr(c, 'phase_init_impl')(self, phase, configs)
        return configs

    def add_phase_module(self, module: 'PhaseInitMixin'):
        self._init_phase_children.append(module)

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        return configs

    def phase_init(self, envs: AllConfigs) -> AllConfigs:
        configs = envs
        init_list = self._phase_init_sequence()
        for p in get_args(PhaseTypes):
            for i in init_list:
                configs = i._phase_init_super(p, configs)
                i._init_phase_done = False
        return configs
