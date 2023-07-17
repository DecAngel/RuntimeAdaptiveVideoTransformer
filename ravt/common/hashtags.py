import hashlib
from typing import Union

import pytorch_lightning as pl


def module_hash(module: Union[pl.LightningModule, pl.LightningDataModule]) -> str:
    return hashlib.md5(f'{module.__class__.__name__}{module.hparams_initial}'.encode('utf-8')).hexdigest()[:6]
