from typing import List, Union
from dataclasses import dataclass
import math
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
    def compute(self) -> MetricOutput:
        raise NotImplementedError("Subclasses must implement compute()")
        
        
class MeanMetric(MetricBase):
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
        
class MAEMetric(MetricBase):
    errors: List[torch.Tensor]
    def __init__(self, dim: int=1):
        super().__init__()
        self.dim = dim
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        error = torch.abs(target - preds)
        error = torch.mean(error, dim=self.dim).flatten()
        self.errors.append(error)
        
    def compute(self) -> MetricOutput:
        errors = torch.cat(self.errors)
        
        if errors.numel() == 0:
            return MetricOutput(0.0, 0.0, 0.0, 0.0, 0.0)
        
        n = errors.numel()
        std = errors.std().item() if n > 1 else 0.0
        return MetricOutput(
            mean=errors.mean().item(),
            std=std,
            std_error=std / (n ** 0.5),
            min=errors.min().item(),
            max=errors.max().item()
        )
        
class MSEMetric(MetricBase):
    errors: List[torch.Tensor]
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        error = torch.pow(target - preds, 2)
        error= torch.mean(error, dim=self.dim).flatten()
        self.errors.append(error)

    def compute(self) -> MetricOutput:
        errors = torch.cat(self.errors)
        if errors.numel() == 0:
            return MetricOutput(0.0, 0.0, 0.0, 0.0, 0.0)

        n = errors.numel()
        std = errors.std().item() if n > 1 else 0.0
        return MetricOutput(
            mean=errors.mean().item(),
            std=std,
            std_error=std / (n ** 0.5),
            min=errors.min().item(),
            max=errors.max().item()
        )
        
class RMSEMetric(MSEMetric):
    def compute(self) -> MetricOutput:
        mse_out = super().compute()
        
        if mse_out.mean == 0:
            return mse_out

        errors = torch.sqrt(torch.cat(self.errors))
        n = errors.numel()
        std = errors.std().item() if n > 1 else 0.0

        return MetricOutput(
            mean=math.sqrt(mse_out.mean),
            std=std,
            std_error=std / (n ** 0.5),
            min=errors.min().item(),
            max=errors.max().item()
        )