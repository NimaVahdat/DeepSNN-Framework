"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    ---------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> None:
        """
        Compute the encoded tensor of the given data.
        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.
        Returns
        -------
        None
            It return the encoded tensor.
        """
        pass


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.time = int(time/dt)
        self.dt = dt
        self.device = device
        self.range_data = kwargs.get("range_data", (0, 255))
        
    def Scale_Data(self) -> None:
        # Complement of data
        self.data = abs(self.data - self.range_data[1] + self.range_data[0])
        # Scaling data to values to time range
        self.data = (self.data - self.range_data[0]) * \
            (self.time - 1) // (self.range_data[1] - self.range_data[0])

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        self.data = data
        self.Scale_Data()
        
        spikes = torch.Tensor(self.time, *self.data.size(), device=self.device)
        for i in range(self.time):
            spikes[i] = self.data == i
        
        return spikes

class PositionEncoder(AbstractEncoder):
    """
    Position coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.time = int(time/dt)
        self.dt = dt
        self.device = device
        self.node_n = kwargs.get("node_n", 10)
        self.range_data = kwargs.get("range_data", (0, 255))
        self.std = kwargs.get("std", 0.5)
        
        # Making gaussian functions represnting our nodes
        self.functions = [self.gaussianFunc(i/self.node_n, self.std) for i in range(self.node_n)]
        
    def gaussianFunc(self, mean, std) -> callable:
        def f(x):
            return (1/(std*np.sqrt(2*np.pi))) * np.e ** ((-1/2)*((x-mean/std )** 2))
        return f

    def __call__(self, data: torch.Tensor) -> None:
        data = data.flatten()
        
        data = data.long() / self.range_data[1]
        
        spikes = torch.zeros(self.node_n, self.time)
        for i in range(len(self.functions)):
            data_ = data.clone()
            
            data_.apply_(self.functions[i])
            data_ = abs(data_ - 1/(self.std*np.sqrt(2*np.pi)))
            
            data_ = data_ * (self.time - 1) * (self.std*np.sqrt(2*np.pi))
            data_.apply_(int)

            s = data_.unique()

            spikes[i][s.tolist()] = 1
        return spikes.t()

class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.time = int(time/dt)
        self.device = device
        
        self.r = kwargs.get("r", 0.06)
        
        
    def __call__(self, data: torch.Tensor) -> None:
        x = torch.randn((self.time, *data.size()), device=self.device).abs()
        x = torch.pow(x, (data * 0.11 + 5) / 50)
        y = torch.tensor(x < self.r * self.dt, dtype=torch.bool, device=self.device)

        return y.view(self.time, *data.shape).byte()
        # time = self.time
        # dt = self.dt
        # device = self.device
        # shape, size = data.shape, data.numel()
        # datum = data.flatten()
        # rate = torch.zeros(size, device=device)
        # rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

        # # Create Poisson distribution and sample inter-spike intervals
        # # (incrementing by 1 to avoid zero intervals).
        # dist = torch.distributions.Poisson(rate=rate, validate_args=False)
        # intervals = dist.sample(sample_shape=torch.Size([time + 1]))
        # intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

        # # Calculate spike times by cumulatively summing over time dimension.
        # times = torch.cumsum(intervals, dim=0).long()
        # times[times >= time + 1] = 0

        # # Create tensor of spikes.
        # spikes = torch.zeros(time + 1, size, device=device).byte()
        # spikes[times, torch.arange(size)] = 1
        # spikes = spikes[1:]

        # return spikes.view(time, *shape)