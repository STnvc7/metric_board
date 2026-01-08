from typing import Literal, List
import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from dsp_board.preprocesses import resample

from metric_board.interface import MetricBase, MetricOutput, MeanMetric
from metric_board.utils.tensor import channelize

class PESQ(MetricBase):
    def __init__(
        self,
        original_sample_rate: int,
        metric_sample_rate: Literal[8000,16000] = 16000,
        mode: Literal["wb", "nb"] = "wb"
    ):
        super().__init__()
        self.original_sample_rate = original_sample_rate
        self.metric_sample_rate = metric_sample_rate
        self.mode = mode
        self.metric = MeanMetric()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = channelize(preds, keep_dims=1) # (..., L) -> (C, L)
        target = channelize(target, keep_dims=1) # (..., L) -> (C, L)
        preds = resample(preds, self.original_sample_rate, self.metric_sample_rate)
        target = resample(target, self.original_sample_rate, self.metric_sample_rate)
        
        try:
            scores = perceptual_evaluation_speech_quality(preds, target, fs=self.metric_sample_rate, mode=self.mode)
            scores = scores.flatten()
            self.metric.update(scores)
        except Exception as e:
            print(f"Error calculating PESQ: {e}")

    def compute(self) -> MetricOutput:
        return self.metric.compute()