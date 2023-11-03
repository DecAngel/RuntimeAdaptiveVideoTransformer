from torch import nn

from ravt.core.constants import BatchDict


class BaseTransform(nn.Module):
    def transform(self, batch: BatchDict) -> BatchDict:
        raise NotImplementedError()
