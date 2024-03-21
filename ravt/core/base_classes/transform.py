from torch import nn

from ..constants import BatchTDict, BatchNDict


class BaseTransform(nn.Module):
    def preprocess_tensor(self, batch: BatchTDict) -> BatchTDict: return batch
    def postprocess_tensor(self, batch: BatchTDict) -> BatchTDict: return batch
    def preprocess_ndarray(self, batch: BatchNDict) -> BatchNDict: return batch
    def postprocess_ndarray(self, batch: BatchNDict) -> BatchNDict: return batch
