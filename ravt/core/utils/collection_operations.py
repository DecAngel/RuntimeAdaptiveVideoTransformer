from typing import Dict, List, Union, Callable, Tuple

import torch
from torch.utils.data import default_collate

from .array_operations import ArrayType


CollectionType = Union[ArrayType, Tuple[ArrayType], List[ArrayType], Dict[str, ArrayType]]


class ApplyCollection:
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, collection: CollectionType) -> CollectionType:
        if isinstance(collection, dict):
            new_dict = {}
            for k, v in collection.items():
                new_dict[k] = self.__call__(v)
            return new_dict
        elif isinstance(collection, list):
            new_list = []
            for v in collection:
                new_list.append(self.__call__(v))
            return new_list
        elif isinstance(collection, tuple):
            new_list = []
            for v in collection:
                new_list.append(self.__call__(v))
            return tuple(new_list)
        else:
            return self.fn(collection)


def get_one_element(collection: CollectionType) -> ArrayType:
    if isinstance(collection, dict):
        return get_one_element(list(collection.values())[0])
    elif isinstance(collection, (list, tuple)):
        return get_one_element(collection[0])
    else:
        return collection


ndarray2tensor = ApplyCollection(torch.from_numpy)
tensor2ndarray = ApplyCollection(lambda t: t.detach().cpu().numpy())
collate = default_collate


def reverse_collate(collection: CollectionType) -> List[CollectionType]:
    batch_size = get_one_element(collection).shape[0]
    return [ApplyCollection(lambda t: t[i])(collection) for i in range(batch_size)]


def select_collate(collection: CollectionType, batch_id: int) -> CollectionType:
    return ApplyCollection(lambda t: t[batch_id])(collection)
