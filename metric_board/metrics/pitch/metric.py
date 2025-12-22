from typing import Literal, List
import math
import torch
from dsp_board.features import pitch

from metric_board.interface import MetricBase, MetricOutput

class Pitch(MetricBase):
    errors: List[torch.Tensor]
    
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
        
        assert distance in ["mae", "mse", "rmse"], f"Unsupported distance: {distance}. Choose from 'mae', 'mse', or 'rmse'."
        self.distance = distance
        
        self.add_state("errors", default=[], dist_reduce_fx="cat")

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
        ).squeeze()
        
        f0 = fn(target)
        f0_preds = fn(preds)
        
        # only voiced frames -----------------------
        nonzero_indeces = torch.logical_and(f0 != 0, f0_preds != 0)
        f0 = f0[nonzero_indeces]
        f0_preds = f0_preds[nonzero_indeces]

        f0 = self.convert_scale(f0)
        f0_preds = self.convert_scale(f0_preds)

        abs_error = torch.abs(f0 - f0_preds)
        self.errors.append(abs_error)
            
        return
    
    def compute(self) -> MetricOutput:
        errors = torch.cat(self.errors, dim=-1).flatten()
        if self.distance == "mae":
            return self.calc_output(errors)
        elif self.distance == "mse":
            return self.calc_output(errors ** 2)
        elif self.distance == "rmse":
            outputs = self.calc_output(errors ** 2)
            rmse = math.sqrt(outputs.mean)
            return MetricOutput(
                mean=rmse,
                std=math.sqrt(outputs.std),
                std_error=outputs.std_error / (2 * rmse) if rmse > 0 else 0.0,
                min=math.sqrt(outputs.min),
                max=math.sqrt(outputs.max)
            )
        else:
            raise ValueError(f"Unsupported distance: {self.distance}. Choose from 'mae', 'mse', or 'rmse'.")