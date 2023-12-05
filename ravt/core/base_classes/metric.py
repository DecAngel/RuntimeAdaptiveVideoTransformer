from torchmetrics import Metric

from ravt.core.constants import BatchTDict, MetricDict


class BaseMetric(Metric):
    def update(self, batch: BatchTDict, pred: BatchTDict) -> None: raise NotImplementedError()

    def compute(self) -> MetricDict: raise NotImplementedError()
