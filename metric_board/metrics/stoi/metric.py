import torch
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

from metric_board.interface import MetricBase, MetricOutput, MeanMetric
from metric_board.utils.tensor import channelize

class STOI(MetricBase):
    def __init__(
        self,
        sample_rate: int,
        extended: bool = True
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.extended = extended
        self.metric = MeanMetric()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = channelize(preds, keep_dims=1) # (..., L) -> (C, L)
        target = channelize(target, keep_dims=1) # (..., L) -> (C, L)
        
        try:
            scores = short_time_objective_intelligibility(preds, target, fs=self.sample_rate, extended=self.extended)
            scores = scores.flatten()
            self.metric.update(scores)
        except Exception as e:
            print(f"Error calculating STOI: {e}")

    def compute(self) -> MetricOutput:
        return self.metric.compute()