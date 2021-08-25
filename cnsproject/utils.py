"""
Module for utility functions.

"""
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple

def format_correct(
        v: torch.Tensor,
        s: torch.Tensor,
        threshold: int = -50,
        v_rest: float = -70
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the input to be plotted in a correct form.
    """
    
    time = torch.Tensor([])
    voltage = torch.Tensor([])
    for i in range(len(v)):
        if s[i] == True:
            voltage = torch.cat((voltage, torch.Tensor([threshold])))
            voltage = torch.cat((voltage, torch.Tensor([v[i]])))
            
            time = torch.cat((time, torch.Tensor([[i], [i]])))
        else:
            voltage = torch.cat((voltage, torch.Tensor([v[i]])))
            time = torch.cat((time, torch.Tensor([[i]])))
    voltage = torch.cat((torch.Tensor([v_rest]), voltage))
    time = torch.cat((torch.Tensor([[0]]), time))   
    return voltage, time

def Lateral_inhibition1(
        picture: torch.Tensor,
        kernel_size: int,
        sigma: float,
        coefficient: float
        ) -> torch.Tensor:
    """
    Applies lateral inhibition on intensities. For each location, this inhibition decreases the intensity of the
	surrounding cells that has lower intensities by a specific factor.

    Parameters
    ----------
    picture : torch.Tensor
        Input tensor to evaluate which attributes to be inhibited.
    kernel_size : int
        Radius of inhibition.
    sigma : float
    coefficient : float
    """
    kernel = np.zeros((kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):
            X = kernel_size // 2 - x
            Y = kernel_size // 2 - y
            kernel[y][x] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(X**2 + Y**2) / (2 * (sigma ** 2))) * (-2 ** coefficient)

    pad = (kernel_size-1) // 2

    filtered = torch.zeros(*picture.shape)
    for feature in range(picture.shape[1]):
        filtered[0][feature] = F.conv2d(input = (picture[0][feature].float()) * torch.ones(1,1,*picture.shape[2:]),
                                        weight = (torch.tensor([[kernel]])).float(), 
                                        bias = torch.zeros(1), 
                                        stride = 1,
                                        padding = [pad, pad])

    f = torch.zeros(*picture.shape)
    for i in range(f.shape[1]):
        f[0][i] = filtered[0][:i].sum(0) + filtered[0][i+1:].sum(0)
    return f

def Lateral_inhibition2(picture, kernel_size, sigma, coefficient):
    kernel = np.zeros((kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):
            X = kernel_size // 2 - x
            Y = kernel_size // 2 - y
            kernel[y][x] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(X**2 + Y**2) / (2 * (sigma ** 2))) * (-2 ** coefficient)

    pad = (kernel_size-1) // 2

    filtered = torch.zeros(*picture.shape)
    for feature in range(picture.shape[1]):
        filtered[0][feature] = F.conv2d(input = (picture[0][feature].float()) * torch.ones(1,1,*picture.shape[2:]),
                                        weight = (torch.tensor([[kernel]])).float(), 
                                        bias = torch.zeros(1), 
                                        stride = 1,
                                        padding = [pad, pad])

 
    return filtered