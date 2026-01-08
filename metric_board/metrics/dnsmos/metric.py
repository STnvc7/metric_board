from typing import List, Literal
import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

from metric_board.interface import MetricBase, MeanMetric, MetricOutput
from metric_board.utils.tensor import channelize

class DNSMOS(MetricBase):
    def __init__(
        self,
        sample_rate: int,
        submetric: Literal["p808", "sig", "bak", "ovr"] = "p808"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.submetric = submetric
        self.submetric_index = {"p808": 0, "sig": 1, "bak": 2, "ovr": 3}[self.submetric]
        self.metric = MeanMetric()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = channelize(preds, keep_dims=1) # (..., L) -> (C, L)
        target = channelize(target, keep_dims=1) # (..., L) -> (C, L)
        
        try:
            scores = deep_noise_suppression_mean_opinion_score(preds, personalized=False, fs=self.sample_rate)
            scores = scores[..., self.submetric_index].flatten()
            self.metric.update(scores)
        except Exception as e:
            print(f"Error calculating DNSMOS: {e}")

    def compute(self) -> MetricOutput:
        return self.metric.compute()
