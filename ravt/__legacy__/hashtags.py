import hashlib
from typing import Any, Union, Iterable, Mapping

import pytorch_lightning as pl


HASH_DEPTH = 2
HASH_WIDTH = 6


def _hash_simple(h, obj: Union[int, bool, str, type]):
    h.update(str(obj).encode('utf-8'))


def _hash_obj(h, obj: Any, depth: int = HASH_DEPTH):
    if depth == 0:
        h.update(b'0')
    _hash_simple(h, obj.__class__)

    if isinstance(obj, (int, bool, str)):
        _hash_simple(h, obj)
    elif isinstance(obj, bytes):
        h.update(obj)
    elif isinstance(obj, pl.LightningModule):
        _hash_obj(h, obj.hparams_initial, depth)
    elif isinstance(obj, Iterable):
        for o in obj:
            _hash_obj(h, o, depth-1)
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _hash_obj(h, k, depth-1)
            _hash_obj(h, v, depth-1)
    elif hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            _hash_obj(h, k, depth - 1)
            _hash_obj(h, v, depth - 1)
    elif hasattr(obj, '__slots__'):
        for k in obj.__slots__:
            _hash_obj(h, k, depth - 1)
            _hash_obj(h, getattr(obj, k), depth - 1)
    return


def hash_all(*args) -> str:
    h = hashlib.md5()
    for a in args:
        _hash_obj(h, a)
    return h.hexdigest()[:HASH_WIDTH]
