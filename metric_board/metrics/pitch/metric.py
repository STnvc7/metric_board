from typing import Literal
import torch
from dsp_board.features import pitch

from metric_board.interface import MetricBase, MetricOutput, MAEMetric, MSEMetric, RMSEMetric
from metric_board.utils.tensor import channelize

class Pitch(MetricBase):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        pitch_extract_method: Literal["dio", "harvest"] = "harvest",
        scale: Literal["linear", "log", "cent"] = "log",
        distance: Literal["mae", "mse", "rmse"] = "mae"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        
        assert pitch_extract_method in ["dio", "harvest"], f"Unsupported pitch extraction method: {pitch_extract_method}. Choose from 'harvest' or 'dio'."
        self.pitch_extract_method: Literal["dio", "harvest"] = pitch_extract_method
        
        assert scale in ["linear", "log", "cent"], f"Unsupported scale: {scale}. Choose from 'linear', 'log', or 'cent'."
        self.scale = scale
        
        if distance == "mae":
            self.metric = MAEMetric(dim=-1)
        elif distance == "mse":
            self.metric = MSEMetric(dim=-1)
        elif distance == "rmse":
            self.metric = RMSEMetric(dim=-1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}. Choose from 'mae', 'mse', or 'rmse'.")
        
    def convert_scale(self, f0: torch.Tensor) -> torch.Tensor:
        if self.scale == "linear":
            return f0
        elif self.scale == "log":
            return torch.log(torch.clamp(f0, min=1e-8))
        elif self.scale == "cent":
            return 1200 * torch.log2(torch.clamp(f0, min=1e-8))
        else:
            raise ValueError(f"Unsupported scale: {self.scale}. Choose from 'linear', 'log', or 'cent'.")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        
        fn = lambda x: pitch(
            x,
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            method=self.pitch_extract_method,
        )
        target = channelize(target, keep_dims=1) #(..., L) -> (C, L)
        preds = channelize(preds, keep_dims=1) #(..., L) -> (C, L)
        f0 = fn(target) #(C, 1, L)
        f0_preds = fn(preds) #(C, 1, L)
        
        # only voiced frames -----------------------
        nonzero_indeces = torch.logical_and(f0 != 0, f0_preds != 0)
        f0 = f0[nonzero_indeces]
        f0_preds = f0_preds[nonzero_indeces]
        f0 = self.convert_scale(f0)
        f0_preds = self.convert_scale(f0_preds)
        self.metric.update(f0_preds, f0)
        return
    
    def compute(self) -> MetricOutput:
        return self.metric.compute()