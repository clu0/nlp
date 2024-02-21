from typing import Tuple
import torch
import torch.nn as nn


def compute_norms(model: nn.Module) -> Tuple[float, float]:
    weight_norm = 0.0
    grad_norm = 0.0
    for p in model.parameters():
        with torch.no_grad():
            weight_norm += p.norm(p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += p.grad.norm(p=2, dtype=torch.float32).item() ** 2
    return weight_norm, grad_norm