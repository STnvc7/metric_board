from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from numpy.typing import DTypeLike


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def to_numpy(x: torch.Tensor, numpy_dtype: Optional[DTypeLike]=None) -> np.ndarray:
    x_np: np.ndarray = x.cpu().detach().clone().numpy()
    if numpy_dtype is not None:
        x_np = x_np.astype(numpy_dtype)
    return x_np

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def from_numpy(x: np.ndarray, device: Optional[torch.device]=None, torch_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    x_torch: torch.Tensor = torch.from_numpy(x)
    if device is not None:
        x_torch = x_torch.to(device)
    if torch_dtype is not None:
        x_torch = x_torch.to(torch_dtype)
    return x_torch

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def fix_length(x: torch.Tensor, length: int, axis=-1) -> torch.Tensor:
    axis = axis if axis >= 0 else x.dim() + axis
    current_length = x.size(axis)

    if current_length > length:
        return x.narrow(axis, 0, length)
    elif current_length < length:
        pad_size = [0] * (2 * x.dim())
        pad_size[2 * (x.dim() - 1 - axis) + 1] = length - current_length

        return F.pad(x, pad_size)
    else:
        return x
        
def channelize(x: torch.Tensor, keep_dims: int) -> torch.Tensor:
    original_shape = x.shape
    if x.ndim < keep_dims:
        raise ValueError(f"Input tensor has {x.ndim} dimensions, but {keep_dims} dimensions are required.")
    elif x.ndim == keep_dims:
        x = x.unsqueeze(0)
    elif x.ndim == keep_dims + 1:
        pass
    else:
        x = x.view(-1, *original_shape[-keep_dims:])
    return x