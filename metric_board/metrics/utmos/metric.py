from typing import List, cast
import torch
from speechmos.utmos22.strong.model import UTMOS22Strong

from metric_board.interface import MetricBase, MetricOutput
from metric_board.utils.tensor import channelize

class UTMOS(MetricBase):
    scores: List[torch.Tensor]
    
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
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = channelize(preds, keep_dims=1) # (..., L) -> (C, L)
        with torch.no_grad():
            predictor = self.predictor.to(preds.device)
            score = predictor(preds, sr=self.sample_rate).flatten()
        self.scores.append(score)

    def compute(self) -> MetricOutput:
        scores = torch.cat(self.scores, dim=-1).flatten()
        return self.calc_output(scores)
