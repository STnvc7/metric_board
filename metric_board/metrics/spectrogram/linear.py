from typing import Literal, List
import math
import torch
from dsp_board.features import spectrogram

from metric_board.interface import MetricBase, MetricOutput, MAEMetric, MSEMetric, RMSEMetric
from metric_board.utils.tensor import channelize

class Spectrogram(MetricBase):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        log: bool = True,
        distance: Literal["mae", "mse", "rmse"] = "mse",
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.log = log
        
        if distance == "mae":
            self.metric = MAEMetric(dim=1)
        elif distance == "mse":
            self.metric = MSEMetric(dim=1)
        elif distance == "rmse":
            self.metric = RMSEMetric(dim=1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}. Choose from 'mae', 'mse', or 'rmse'.")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        
        fn = lambda x: spectrogram(
            x,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            log=self.log,
        )
        
        target = channelize(target, keep_dims=1) #(..., L) -> (C, L)
        preds = channelize(preds, keep_dims=1) #(..., L) -> (C, L)
        spc_target = fn(target) #(C, F, T)
        spc_preds = fn(preds) #(C, F, T)
        
        self.metric.update(spc_preds, spc_target)
        return

    def compute(self) -> MetricOutput:
        return self.metric.compute()