from torchmetrics.detection import MeanAveragePrecision

from ravt.protocols.structures import MetricDict, BatchDict, PredDict
from ravt.protocols.classes import BaseMetric


class TorchMetricsMAPMetric(BaseMetric):
    # TODO: finish
    def __init__(self):
        super().__init__()
        self.map = MeanAveragePrecision()

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        self.

    def update(self, batch: BatchDict, pred: PredDict) -> None:


    def compute(self) -> MetricDict:
        pass