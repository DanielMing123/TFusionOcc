import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from typing import NamedTuple

SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

def softmax_inverse_with_grad(p, mode='zero_mean', eps=1e-8):
    """
    Inverse softmax (up to a constant shift) for B×N×C tensor.
    
    Args:
        p: Tensor of shape (B, N, C) — softmax output
        mode: 'zero_mean' or 'zero_reference'
        eps: Small constant to avoid log(0)
        
    Returns:
        Tensor of shape (B, N, C), inverse logits (up to a constant)
    """
    log_p = torch.log(p + eps)  # safe log

    if mode == 'zero_mean':
        mean = log_p.mean(dim=-1, keepdim=True)  # shape: (B, N, 1)
        z = log_p - mean
    elif mode == 'zero_reference':
        ref = log_p[..., [0]]  # shape: (B, N, 1), using first class as reference
        z = log_p - ref
    else:
        raise ValueError("Invalid mode. Use 'zero_mean' or 'zero_reference'.")
    
    return z

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))

def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]

class PrimitivePrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    u: Tensor = None
    v: Tensor = None
    deform_weights: Tensor = None
    semantics: Tensor = None
    original_means: Tensor = None
    delta_means: Tensor = None