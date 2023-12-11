import contextlib
from pathlib import Path
from typing import Dict, Union, List, Tuple

import torch

from ..utils.lightning_logger import ravt_logger as logger


class CheckpointMixin:
    # device: torch.device
    # def state_dict(self) -> Dict: ...
    # def load_state_dict(self, state_dict: Dict, strict: bool = False) -> Tuple[List[str], List[str]]: ...

    def pth_adapter(self, state_dict: Dict) -> Dict: ...

    def load_from_pth(self, file_path: Union[str, Path]) -> None:
        state_dict = torch.load(str(file_path), map_location='cpu')
        with contextlib.suppress(NotImplementedError):
            state_dict = self.pth_adapter(state_dict)

        misshaped_keys = []
        ssd = self.state_dict()
        for k in list(state_dict.keys()):
            if k in ssd and ssd[k].shape != state_dict[k].shape:
                misshaped_keys.append(k)
                del state_dict[k]

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = list(filter(lambda key: key not in misshaped_keys, missing_keys))

        if len(missing_keys) > 0:
            logger.warning(f'Missing keys in ckpt: {missing_keys}')
        if len(unexpected_keys) > 0:
            logger.warning(f'Unexpected keys in ckpt: {unexpected_keys}')
        if len(misshaped_keys) > 0:
            logger.warning(f'Misshaped keys in ckpt: {misshaped_keys}')
        logger.info(f'pth file {file_path} loaded!')

    def load_from_ckpt(self, file_path: Union[str, Path], strict: bool = True):
        self.load_state_dict(
            torch.load(
                str(file_path),
                map_location=self.device
            )['state_dict'],
            strict=strict,
        )
        logger.info(f'ckpt file {file_path} loaded!')
