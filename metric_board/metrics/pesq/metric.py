from typing import Literal, List
import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from dsp_board.preprocesses import resample

from metric_board.interface import MetricBase, MetricOutput
from metric_board.utils.tensor import channelize

class PESQ(MetricBase):
    scores: List[torch.Tensor]
    
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
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = channelize(preds, keep_dims=1) # (..., L) -> (C, L)
        target = channelize(target, keep_dims=1) # (..., L) -> (C, L)
        preds = resample(preds, self.original_sample_rate, self.metric_sample_rate)
        target = resample(target, self.original_sample_rate, self.metric_sample_rate)
        
        try:
            score = perceptual_evaluation_speech_quality(preds, target, fs=self.metric_sample_rate, mode=self.mode)
            self.scores.append(score)
        except Exception as e:
            print(f"Error calculating PESQ: {e}")

    def compute(self) -> MetricOutput:
        scores = torch.cat(self.scores, dim=-1).flatten()
        return self.calc_output(scores)
