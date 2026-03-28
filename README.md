# metric_board

## Overview
- metric_board is a library designed for the objective evaluation of synthesized speech.
- It is built by inheriting from torchmetrics, making it highly compatible and easy to integrate with PyTorch and PyTorch Lightning workflows.
- The compute method provides a comprehensive statistical breakdown, calculating not only the mean but also the standard deviation, standard error, maximum, and minimum values.
- By utilizing the Evaluator class, the library supports multiple input types, including torch.Tensor, np.ndarray, and direct file paths.

## Installation
You can install the package using [uv](https://docs.astral.sh/uv/) or [pip](https://pypi.org/project/pip/).
```bash
# uv
uv add https://github.com/STnvc7/metric_board.git
# pip
pip install https://github.com/STnvc7/metric_board.git
```

## Available Metrics
- DNSMOS (Deep Noise Surpression MOS)
- MCD (Mel Cepstral Distortion)
- PESQ (Perceptural Evaluation of Speech Quality)
- Pitch
- Spectrogram
- MelSpectrogram
- STOI (Short-Time Objective Intelligibility)
- UTMOS
- VUV

## Usage
### Example1: Evaluator
```python
from objeva.evaluator import Evaluator
from objeva.metrics import MCD, PESQ

sample_rate = 44100
fft_size = 1024
hop_size = 256

# Define the metrics you want to use (e.g., MCD, PESQ) with the appropriate parameters.
metrics = {
    "mcd": MCD(sample_rate, fft_size, hop_size),
    "pesq": PESQ(sample_rate, mode="wb"),
}

# Create an `Evaluator` instance with your metrics.
evaluator = Evaluator(metrics)
```

### Example2: List evaluation using evaluate method
```python
# Prepare lists of prediction and target audio file paths or tensor.
preds_path = Path("./test/generated")
target_path = Path("./test/ground_truth")
preds = sorted(preds_path.glob("*.wav"))
target = sorted(target_path.glob("*.wav"))

# Call the `evaluate` method with the prediction and target lists to evaluate all pairs at once.
results = evaluator.evaluate(preds, target)
```

### Example3: Step-by-step evaluation using update and compute method
```python
# For scenarios such as model training or streaming inference, you can evaluate each prediction-target pair one by one.
for batch in dataloader:
    y = batch["y"]
    y_hat = model(batch["x"])
    
    # Call the update method for each pair as you process them (e.g., inside a loop).
    evaluator.update(y_hat, y)

# After all pairs have been processed, call the compute method to get the final results for each metric.
result = evaluator.compute()
```

## Custom Metric
- You can use your own custom metrics by passing any class that inherits from `metric_board.MetricBase` to the `Evaluator`.  

## License
This repository is licensed under the MIT License.