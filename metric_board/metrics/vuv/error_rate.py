from typing import Literal, List
import torch
from dsp_board.features import vuv as extract_vuv

from metric_board.interface import MetricBase, MetricOutput
from metric_board.utils.tensor import channelize

class VUVErrorRate(MetricBase):
    values: List[torch.Tensor]
    
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
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        vuv = extract_vuv(target, self.sample_rate, self.hop_size, self.pitch_extract_method)
        vuv_preds = extract_vuv(preds, self.sample_rate, self.hop_size, self.pitch_extract_method)
        acc = vuv != vuv_preds
        acc = acc.sum() / acc.numel()
        acc = acc.flatten()
        self.values.append(acc)
        return
    
    def compute(self) -> MetricOutput:
        values = torch.cat(self.values, dim=-1).flatten()
        return self.calc_output(values)
