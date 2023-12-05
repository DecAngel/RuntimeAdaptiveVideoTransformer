from torch import nn

from ..constants import BatchTDict, BatchNDict


class BaseTransform(nn.Module):
    def postprocess_tensor(self, batch: BatchTDict) -> BatchTDict: raise NotImplementedError()

    def preprocess_tensor(self, batch: BatchTDict) -> BatchTDict: raise NotImplementedError()

    def postprocess_ndarray(self, batch: BatchNDict) -> BatchNDict: raise NotImplementedError()

    def preprocess_ndarray(self, batch: BatchNDict) -> BatchNDict: raise NotImplementedError()
