from typing import List, get_args

from ..structures import InternalConfigs, ConfigTypes


class PhaseInitMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_phase_done: bool = False
        self._init_phase_children: List[PhaseInitMixin] = []

    def __setattr__(self, key, value):
        if isinstance(value, PhaseInitMixin):
            self._init_phase_children = value
        return super().__setattr__(key, value)

    def _phase_init_sequence(self) -> List['PhaseInitMixin']:
        if self._init_phase_done:
            return []
        else:
            self._init_phase_done = True

        res = [self]
        res.extend([c._phase_init_sequence() for c in self._init_phase_children])
        return res

    def _phase_init_super(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        s = super()
        if isinstance(s, PhaseInitMixin):
            configs = s._phase_init_super(phase, configs)

        try:
            configs = self.phase_init_impl(phase, configs)
        except NotImplementedError:
            pass
        return configs

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        raise NotImplementedError()

    def phase_init(self, configs: InternalConfigs) -> InternalConfigs:
        init_list = self._phase_init_sequence()
        for p in get_args(ConfigTypes):
            for i in init_list:
                configs = i._phase_init_super(p, configs)
        return configs
