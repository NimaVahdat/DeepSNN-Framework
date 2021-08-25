"""
Module for visualization and plotting.

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

from typing import Tuple, Optional, Callable

from ..utils import format_correct


def plot_voltage(
        v: torch.Tensor,
        s: torch.Tensor,
        threshold: float,
        v_rest: float,
        dt: int,
        time: Optional[Tuple[int, int]] = None,
        theta_rh: Optional[float] = None
) -> None:
    """
    Plot voltage for a single neuron on a specific example.
    
    Parameters
    ----------
    v : torch.Tensor
        Recorded Voltages.
    s : torch.Tensor
        Spike occurrences.
    threshold : float
        Spike threshold voltage.
    v_rest : float
        Resting membrane voltage.
    dt : int
        Time resolution.
    time : Optional[Tuple[int, int]], optional
        Interval time to be displayed. The default is None.

    Returns
    -------
    None

    """
    voltages, times = format_correct(v, s, threshold, v_rest)
    
    # Checking input time
    if time is None:
        time = (0, v.shape[0])
    else:
        assert time[0]/dt < time[1]
        assert time[1]/dt <= v.shape[0]
    
    time_ticks = np.arange(time[0], time[1] + 1, dt)    
    
    plt.figure()
    plt.plot(times, voltages, color = "green")
    plt.xlabel("Time")
    plt.ylabel("Voltage")      
    locs, labels = plt.xticks()
    locs = range(int(locs[1]), int(locs[-1]), 10)
    plt.xticks(locs, time_ticks)

    # Draw threshold
    plt.axhline(threshold, linestyle="--", color="blue", zorder=0)
        
    # Draw v_rest line
    plt.axhline(v_rest, linestyle="--", color="black", zorder=0)
    
    if theta_rh != None:
        plt.axhline(theta_rh, linestyle="--", color="gray", zorder=0)
    
    plt.show()

    
def plot_current(
        current: Callable[[int], int],
        time: Tuple[int, int],
        dt: int = 1,
        label: Optional[str] = "",
) -> None:
    """
    Plot current.

    Parameters
    ----------
    current : function
        Current function that displays the current in time.
    time : Tuple[int, int]
        Interval time to be displayed.
    dt : int, optional
        Time resolution. The default is 1.

    Returns
    -------
    None

    """
    
    # setting the time - coordinates
    t = np.linspace(time[0], time[1], (time[1] - time[0])//dt)
    plt.figure(figsize=(20,5)) 
    plt.xlim(0, time[1]-time[0])
    plt.ylim(0, 120)

    # setting the corresponding current - coordinates
    c = np.frompyfunc(current, 1, 1)
    
    plt.xlabel("Time")
    plt.ylabel("]") 
    plt.title(label)
    
    # potting the points
    plt.plot(t, c(t).astype(np.int), markersize=50)
      
    plt.show()    

    
def plot_F_I(
        I: torch.Tensor,
        F: torch.Tensor
) -> None:
    """
    Plot F-I curve.

    Parameters
    ----------
    I : torch.Tensor
        ]nput currents.
    F : torch.Tensor
        Frequency of spikes for input currents.

    Returns
    -------
    None

    """
    plt.plot(I, F, color="red")
    plt.xlabel("I(t)")
    plt.ylabel("f=1/T")
    plt.title("Frequency-Current relation")
    plt.show()
    
def raster(
        s_exc: torch.Tensor = [],
        s_inh: torch.Tensor = [],
        label: Optional[str] = "",
        ):
    plt.figure(figsize=(15,5))
    plt.xlim(0, len(s_exc))
    if s_exc != []:
        exc = torch.nonzero(s_exc)
        plt.scatter(exc[:,0], exc[:,1], s=1, c='darkviolet')
    if s_inh != []:
        inh = torch.nonzero(s_inh)
        if s_exc != []:
            inh[:,1] += s_exc.size(1)
        plt.scatter(inh[:,0], inh[:,1], s=1, c='fuchsia')
        
    plt.xlabel("Time")
    plt.ylabel("Neurons")
    plt.title("Raster Plot "+label)
    plt.show()
    
def population_activity(
        s: torch.Tensor,
        label: str = ""):
    
    if "exc" in label:
        color = "darkviolet"
    elif "inh" in label:
        color = "fuchsia"
    
    plt.figure(figsize=(20,5)) 
    plt.xlim(0, len(s))
    ss = s.sum(1) / len(s[0])
    time = [i for i in range(len(ss))]
    
    plt.plot(time, ss, c=color)
    
    plt.xlabel("Time")
    plt.title("Population Activity " + label)    
    plt.show()
    
def plot_w(
        w: torch.Tensor,
        wmin: float = 0,
        wmax: float = 2,
        ):

    pre_size = len(w[0])
    post_size = len(w[0][0])
    
    fig, axs = plt.subplots(post_size, figsize=(40,20))
    fig.suptitle("Weight changes", size=79)
    time = np.arange(0, len(w), 1)
    
    for i in range(post_size):
        for j in range(pre_size):
            x = w[:, j, i]
            axs[i].plot(time, x)
            axs[i].set(xlabel="W", ylabel="Time")
     
    plt.show()
    
def plot_reward(d: torch.Tensor):
    plt.figure(figsize=(15,5))
    plt.xlabel("Time")
    plt.title("Reward")
    time = np.arange(0, len(d), 1)
    plt.plot(time, d, color="gold")
    plt.show()