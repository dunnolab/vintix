import math
from typing import List

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/rmsnorm.py
    """

    def __init__(self,
                 size: int,
                 dim: int = -1,
                 eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class GptNeoxMLP(nn.Module):
    """
    https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/model.py
    """

    def __init__(self,
                 hidden_dim: int,
                 intermediate_size: int,
                 bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, intermediate_size, bias=bias)
        self.proj = nn.Linear(intermediate_size, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.fc(x)
        x = nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    """
    https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/model.py
    """

    def __init__(self,
                 hidden_dim: int,
                 intermediate_size: int,
                 bias: bool = True):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, intermediate_size, bias)
        self.fc_2 = nn.Linear(hidden_dim, intermediate_size, bias)
        self.proj = nn.Linear(intermediate_size, hidden_dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


def get_norm_class(norm_class: str) -> nn.Module:
    """Get class for normalization layer"""
    if norm_class == "LayerNorm":
        return nn.LayerNorm
    elif norm_class == "RMSNorm":
        return RMSNorm
    else:
        raise Exception(f"ValueError: {norm_class} is unknown norm type")


def get_mlp_class(mlp_class: str) -> nn.Module:
    """Get class for MLP layer"""
    if mlp_class == "LLaMAMLP":
        return LLaMAMLP
    elif mlp_class == "GptNeoxMLP":
        return GptNeoxMLP
    else:
        raise Exception(f"ValueError: {mlp_class} is unknown mlp type")


def get_alibi_slopes(n: int) -> List[float]:
    """Get ALiBI slopes"""
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(
                2 * closest_power_of_2
            )[0::2][: n - closest_power_of_2]
        )
