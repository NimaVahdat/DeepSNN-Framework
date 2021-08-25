"""
Module for decision making.

"""
import torch
from typing import Union, Sequence, Iterable, Optional
from abc import ABC, abstractmethod

import numpy as np

class AbstractDecision(ABC):
    """
    Abstract class to define decision making strategy.

    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It returns the decision result.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the variables after making the decision.

        Returns
        -------
        None

        """
        pass

class WinnerTakeAllDecision():
    """
    The k-Winner-Take-All decision mechanism.
    
    """
    def __init__(
        self,
        k: int,
        shape: Iterable[int],
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.k = k
        self.shape = shape
        self.chosen = torch.zeros(*shape)

    def compute(self, spikes, voltage) -> None:
        """
        Infer the decision to be made.
        Returns
        -------
        None
            It returns the decision result.
        """     
        if ((self.k == 0) or (spikes.sum() == 0)):
            return torch.zeros(*self.shape)
        real_spikes = spikes * (1 - self.chosen)
        real_voltage = voltage - self.chosen * (abs(torch.max(voltage)) + abs(torch.min(voltage)))
        sum_spikes = real_spikes.sum(3).sum(2).squeeze() 
        sum_spikes_bool = (sum_spikes >= 1).float()
        if (sum_spikes_bool.sum() <= self.k):
            result = torch.zeros(*self.shape)
            for i in range(self.shape[1]):
                if (sum_spikes[i] > 1):
                    result[0][i] = (real_voltage[0][i] >= real_voltage[0][i].max()).float()
                    self.chosen[0][i] = torch.ones(self.chosen.shape[2], self.chosen.shape[3])
                elif (sum_spikes[i] == 1):
                    result[0][i] = real_spikes[0][i]
                    self.chosen[0][i] = torch.ones(self.chosen.shape[2], self.chosen.shape[3])
            self.k -= int(sum_spikes_bool.sum())
        else:
            flatten_voltage = torch.flatten(real_voltage, 2, 3)
            result = torch.zeros(*flatten_voltage.shape)
            k_winner_value = (torch.topk(input = flatten_voltage, k = 1)[0]).flatten()
            k_winner_index = (torch.topk(input = flatten_voltage, k = 1)[1]).flatten()
            index = list(range(0, real_voltage.shape[1]))
            zipped = zip(k_winner_value, index)
            index = [x for _, x in sorted(zipped, reverse=True)]
            for i in range(self.k):
                result[0][index[i]][k_winner_index[index[i]]] = 1
            result = result.view(*self.shape)
            self.k = 0
        return result
