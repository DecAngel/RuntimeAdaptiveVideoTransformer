from torchmetrics.detection import MeanAveragePrecision

from ravt.core.constants import MetricDict, BatchDict, PredDict, PhaseTypes, AllConfigs
from ravt.core.base_classes import BaseMetric


class TorchMetricsMAPMetric(BaseMetric):
    # TODO: finish
    def __init__(self):
        super().__init__()
        self.map = MeanAveragePrecision()

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        pass

    def update(self, batch: BatchDict, pred: PredDict) -> None:
        pass

    def compute(self) -> MetricDict:
        pass