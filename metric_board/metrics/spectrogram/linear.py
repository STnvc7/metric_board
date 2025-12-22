from typing import Literal, List
import math
import torch
from dsp_board.features import spectrogram

from metric_board.interface import MetricBase, MetricOutput
from metric_board.utils.tensor import channelize

class Spectrogram(MetricBase):
    errors: List[torch.Tensor]
    
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        power: bool = False,
        log: bool = True,
        distance: Literal["mae", "mse", "rmse"] = "mse",
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.power = power
        self.log = log
        
        assert distance in ["mae", "mse", "rmse"], f"Unsupported distance: {distance}. Choose from 'mae', 'mse', or 'rmse'."
        self.distance = distance
        
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        
        fn = lambda x: spectrogram(
            x,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            power=self.power,
            log=self.log,
        )
        
        target = channelize(target, keep_dims=1) #(..., L) -> (C, L)
        preds = channelize(preds, keep_dims=1) #(..., L) -> (C, L)
        spc_target = fn(target) #(C, F, T)
        spc_preds = fn(preds) #(C, F, T)
        
        abs_error = torch.abs(spc_target - spc_preds)
        frame_error = torch.mean(abs_error, dim=1).flatten()
        self.errors.append(frame_error)
            
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