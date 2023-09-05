from torchmetrics.detection import MeanAveragePrecision

from ravt.protocols.structures.batch import MetricDict, BatchDict
from .base import BaseMetric


class TorchMetricsMAPMetric(BaseMetric, MeanAveragePrecision):
    # TODO: finish
    def update(self, batch: BatchDict, pred: PredDict) -> None:
        super(MeanAveragePrecision, self).update()

    def compute(self) -> MetricDict:
        pass