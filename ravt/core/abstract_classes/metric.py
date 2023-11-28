from torchmetrics import Metric

from ravt.core.constants import BatchDict, PredDict, MetricDict


class BaseMetric(Metric):
    def update(self, batch: BatchDict, pred: PredDict) -> None:
        raise NotImplementedError()

    def compute(self) -> MetricDict:
        raise NotImplementedError()
