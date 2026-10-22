"""This module bundles configs for creating optimizers and schedulers in a CLI-friendly way.

The main issue we have is that different optimizers/schedules require different arguments. This
would require using **kwargs in the `LightningModule` to accept varying arguments. However, this is
not accepted by the CLI.  
In theory, both optimizer and schedule can also be configured directly in the config, rather than
by the `LightningModule`. But this is limited: For example, the number of total training steps is
not yet known if we use `max_epochs` for training, so we cannot properly configure the scheduler.
Even if we used `max_steps` instead of `max_epochs`, we would not be able to use things like a
fraction for warmup. Using these config classes, and then `configure_optimizers` in the
`LightningModule` gives us the most flexibility and ease of use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

import torch
from torch import nn


class OptimizerConfig(ABC):
    """Base class for bundling parameters for and constructing an optimizer."""
    @abstractmethod
    def build(self,
              parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        """Construct and return the optimizer object.
        
        Parameters:
            parameters: Model parameters to optimize.
        """


@dataclass(frozen=True)
class AdamWConfig(OptimizerConfig):
    """Config for Adam with decoupled weight decay."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    def build(self,
              parameters: Iterable[nn.Parameter]) -> torch.optim.AdamW:
        return torch.optim.AdamW(parameters,
                                 lr=self.lr,
                                 weight_decay=self.weight_decay,
                                 betas=self.betas,
                                 eps=self.eps,
                                 amsgrad=self.amsgrad)


class SchedulerConfig(ABC):
    """Base class for constructing a learning rate schedule."""
    @abstractmethod
    def build(self,
              optimizer: torch.optim.Optimizer,
              n_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
        """Construct and return the schedule object.
        
        Parameters:
            optimizer: torch optimizer instance.
            n_steps: Total number of steps for the schedule.
        """
        pass


@dataclass(frozen=True)
class CosineWarmupConfig(SchedulerConfig):
    """Config for cosine learning rate decay with linear warmup."""
    warmup_fraction: float = 0.05

    def build(self,
              optimizer: torch.optim.Optimizer,
              n_steps: int) -> CosineLRWarmup:
        return CosineLRWarmup(optimizer,
                              n_steps=n_steps,
                              warmup=self.warmup_fraction)


@dataclass(frozen=True)
class OptimizationConfig:
    """Config bundling an optimizer and corresponding learning rate schedule."""
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    def build(self,
              parameters: Iterable[nn.Parameter],
              n_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Construct and return optimizer and schedule objects.
        
        Parameters:
            parameters: Model parameters to optimize.
            n_steps: Total number of steps for the learning rate schedule.
        """
        optimizer = self.optimizer.build(parameters)
        scheduler = self.scheduler.build(optimizer, n_steps)

        return optimizer, scheduler
    

class CosineLRWarmup(torch.optim.lr_scheduler.SequentialLR):
    """Combines linear warmup with subsequent cosine decay.
            
    TODO could create a more general "warmup + other decay" wrapper.
    """
    @override
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 n_steps: int,
                 warmup: float):
        """Create the schedule.

        Parameters:
            optimizer: torch optimizer object to apply the schedule to.
            n_steps: Total number of steps for the schedule (warmup + decay). You are responsible
                for correctly figuring out whether this should be *training steps* or *epochs*. For
                our purposes in Lightning, it's most likely steps, not epochs.
            warmup: Proportion of training that should be spent on warmup.
        """
        self.warmup_steps = int(n_steps * warmup)
        self.decay_steps = n_steps - self.warmup_steps
        self.total_steps = n_steps

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                   start_factor=1/self.warmup_steps,
                                                   total_iters=self.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.decay_steps)
        super().__init__(optimizer,
                         schedulers=[warmup, cosine],
                         milestones=[self.warmup_steps])
