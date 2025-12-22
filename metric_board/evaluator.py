from typing import Dict, Iterable, Union
from os import PathLike
from tqdm import tqdm
import torch
import torchaudio
import numpy as np

from metric_board.interface import MetricBase, MetricOutput

class Evaluator:
    """
    A class for evaluating predictions against targets using multiple metrics.

    This class supports flexible input types (file path, numpy.ndarray, torch.Tensor) and
    manages evaluation, error counting, and logging for each metric.
    """
    def __init__(
        self,
        metrics: Dict[str, MetricBase],
    ):
        self.metrics = metrics
        self.n_evaluations = 0
        self.error_counts = {k: 0 for k in metrics.keys()}
    
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def evaluate(
        self,
        preds_list: Iterable[Union[torch.Tensor,str,PathLike]],
        target_list: Iterable[Union[torch.Tensor,str,PathLike]],
    ) -> Dict[str, MetricOutput]:
        """
        Evaluate predictions against targets using the registered metrics.

        Args:
            preds_list (List[Union[torch.Tensor, str, PathLike]]): 
                List of prediction data. Each element can be a torch.Tensor, file path, or numpy.ndarray.
            target_list (List[Union[torch.Tensor, str, PathLike]]): 
                List of target data. Each element can be a torch.Tensor, file path, or numpy.ndarray.

        Raises:
            ValueError: If the lengths of preds_list and target_list do not match.

        Returns:
            Dict[str, float]: Dictionary of metric name to computed value.
        """
        preds_list = list(preds_list)
        target_list = list(target_list)

        if len(preds_list) != len(target_list):
            raise ValueError(
                f"Length mismatch: preds({len(preds_list)}) and target({len(target_list)}) must have the same length."
            )
        self.n_evaluations = len(preds_list) 
        
        for preds, target in tqdm(zip(preds_list, target_list), total=self.n_evaluations):
            self.update(preds, target)
        
        results = self.compute()
        return results

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _input_to_tensor(self, x):
        if isinstance(x, (str, PathLike)):
            audio, _ = torchaudio.load(x, normalize=True)
            audio = audio.squeeze()
        elif isinstance(x, np.ndarray):
            audio = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            audio = x
        else:
            raise ValueError(
                f"Unsupported input type: {type(x)}. "
                "Input must be a file path, numpy.ndarray, or torch.Tensor."
            )
        return audio
    
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def update(self, preds: Union[torch.Tensor, str, PathLike], target: Union[torch.Tensor, str, PathLike]):
        """
        Update all metrics with a single prediction-target pair.

        Args:
            preds (torch.Tensor): Prediction tensor.
            target (torch.Tensor): Target tensor.
        """
        preds = self._input_to_tensor(preds)
        target = self._input_to_tensor(target)
        
        for key, metric in self.metrics.items():
            try:
                metric.update(preds, target)
            except Exception as e:
                self.error_counts[key] += 1
                print(f"Error in metric '{key}': {type(e).__name__}: {e}\n")
        return
    
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def compute(self) -> Dict[str, MetricOutput]:
        result = {}
        for key, metric in self.metrics.items():
            result[key] = metric.compute()
        return result

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
        self.n_evaluations = 0
        self.error_counts = {k: 0 for k in self.error_counts}
        return
