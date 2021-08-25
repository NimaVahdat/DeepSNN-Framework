"""
Module for reward dynamics.

"""

from abc import ABC, abstractmethod
from typing import Callable
import torch

class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass

class Reward(AbstractReward):
    """
    """
    def __init__(
            self,
            DA: Callable = None,
            tau_d: torch.Tensor = 10,
            dt: torch.Tensor = 1.0,
            **kwarges
    ) -> None:
        self.DA = DA
        self.tau_d = tau_d
        self.dt = dt
        self.d = 0
        
    def compute(self, in_pattern, out_pattern, **kwargs) -> torch.Tensor:
        da = self.DA(in_pattern, out_pattern)
        dd_dt = -(self.d/self.tau_d) + da
        self.d += dd_dt * self.dt
        
        return self.d
    
    def update(self, **kwargs) -> None:
        pass