from typing import cast
import torch
from speechmos.utmos22.strong.model import UTMOS22Strong

from metric_board.interface import MetricBase, MetricOutput, MeanMetric
from metric_board.utils.tensor import channelize

class UTMOS(MetricBase):
    def __init__(
        self,
        sample_rate: int,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.predictor = cast(UTMOS22Strong, torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong", trust_repo=True))
        self.predictor.eval()
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.metric = MeanMetric()
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = channelize(preds, keep_dims=1) # (..., L) -> (C, L)
        with torch.inference_mode():
            predictor = self.predictor.to(preds.device)
            score = predictor(preds, sr=self.sample_rate).flatten()
        self.metric.update(score)

    def compute(self) -> MetricOutput:
        return self.metric.compute()