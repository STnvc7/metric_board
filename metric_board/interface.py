from dataclasses import dataclass
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

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError("Subclasses must implement update(preds, target)")

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