import functools
import math

import torch
from torch import optim


# source:  https://gist.github.com/akshaychawla/86d938bc6346cf535dce766c83f743ce
def _cosine_decay_warmup(iteration: int,
                         warmup_iterations: int,
                         total_iterations: int) -> float:
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / \
            (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _constant_warmup(iteration: int,
                     warmup_iterations: int) -> float:
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def cosine_annealing_with_warmup(optimizer: optim.Optimizer,
                                 warmup_steps: int,
                                 total_steps: int
                                 ) -> optim.lr_scheduler.LambdaLR:
    """Cosine annealing scheduler"""
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def linear_warmup(optimizer: optim.Optimizer,
                  warmup_steps: int
                  ) -> optim.lr_scheduler.LambdaLR:
    """Linear scheduler"""
    _decay_func = functools.partial(
        _constant_warmup,
        warmup_iterations=warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
