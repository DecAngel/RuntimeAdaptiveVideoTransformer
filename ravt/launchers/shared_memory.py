from typing import Optional

from ravt.core.base_classes import BaseLauncher, launcher_entry
from ravt.core.constants import AllConfigs
from ravt.core.functional_classes import SharedMemoryServer

from ..base_configs import environment_configs


class SharedMemoryLauncher(BaseLauncher):
    def __init__(self, envs: Optional[AllConfigs] = None,):
        super().__init__(envs or environment_configs)
        self.server = SharedMemoryServer()

    @launcher_entry()
    def run(self):
        self.server.run()
