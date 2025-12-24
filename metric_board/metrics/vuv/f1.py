from typing import Literal, List
import torch
from torchmetrics.functional.classification import binary_f1_score
from dsp_board.features import vuv as extract_vuv

from metric_board.interface import MetricBase, MetricOutput
from metric_board.utils.tensor import channelize

class VUVF1(MetricBase):
    scores: List[torch.Tensor]
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        pitch_extract_method: Literal["dio", "harvest"]="harvest",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        assert pitch_extract_method in ["dio", "harvest"], f"Unsupported pitch extraction method: {pitch_extract_method}. Choose from 'harvest' or 'dio'."
        self.pitch_extract_method: Literal["dio", "harvest"] = pitch_extract_method
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        vuv = extract_vuv(target, self.sample_rate, self.hop_size, self.pitch_extract_method)
        vuv_preds = extract_vuv(preds, self.sample_rate, self.hop_size, self.pitch_extract_method)
        score = binary_f1_score(vuv, vuv_preds, multidim_average="samplewise")
        score = score.flatten()
        self.scores.append(score)
        return
    
    def compute(self) -> MetricOutput:
        f1 = torch.cat(self.scores)
        return self.calc_output(f1)