from typing import Literal, Optional
import torch
from torchmetrics import Metric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from dsp_board.preprocesses import resample
from dsp_board.processor import Processor

class PESQ(Metric):
    def __init__(
        self,
        original_sample_rate: int,
        target_sample_rate: Literal[8000,16000] = 16000,
        mode: Literal["wb", "nb"] = "wb",
        dsp_processor: Optional[Processor] = None,
    ):
        super().__init__()
        self.original_sample_rate = original_sample_rate
        self.target_sample_rate = target_sample_rate
        self.pesq = PerceptualEvaluationSpeechQuality(fs=target_sample_rate, mode=mode)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = resample(preds, self.original_sample_rate, self.target_sample_rate)
        target = resample(target, self.original_sample_rate, self.target_sample_rate)
        self.pesq.update(preds, target)

    def compute(self) -> torch.Tensor:
        return self.pesq.compute()
    
    def reset(self):
        self.pesq.reset()
        super().reset()