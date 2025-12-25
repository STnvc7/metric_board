from typing import List, Union
from dataclasses import dataclass
from numpy import int128
from torchmetrics import Metric
import torch

@dataclass
class MetricOutput:
    mean: float
    std: float = 0.0
    std_error: float = 0.0
    min: float = 0.0
    max: float = 0.0
    
class MetricBase(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError("Subclasses must implement update()")
        
    def compute(self) -> MetricOutput:
        raise NotImplementedError("Subclasses must implement compute()")
        
    def calc_output(self, values: torch.Tensor) -> MetricOutput:
        if values.numel() == 0:
            return MetricOutput(0.0, 0.0, 0.0, 0.0, 0.0)
        
        mean_val = torch.mean(values).item()
        std_val = torch.std(values).item() if values.numel() > 1 else 0.0
        n = values.numel()
        
        return MetricOutput(
            mean=mean_val,
            std=std_val,
            std_error=std_val / (n ** 0.5),
            min=torch.min(values).item(),
            max=torch.max(values).item()
        )
        
class MeanMetric(Metric):
    values: List[torch.Tensor]
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")
        
    def update(self, value: Union[torch.Tensor, float, int]):
        if isinstance(value, (float, int)):
            value = torch.tensor([float(value)])
        self.values.append(value)
        
    def compute(self) -> MetricOutput:
        values = torch.cat(self.values)
        
        if values.numel() == 0:
            return MetricOutput(0.0, 0.0, 0.0, 0.0, 0.0)
        
        mean_val = torch.mean(values).item()
        std_val = torch.std(values).item() if values.numel() > 1 else 0.0
        n = values.numel()
        
        return MetricOutput(
            mean=mean_val,
            std=std_val,
            std_error=std_val / (n ** 0.5),
            min=torch.min(values).item(),
            max=torch.max(values).item()
        )