import torch
from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor

from cnsproject.plotting.plotting import plot_voltage
from cnsproject.plotting.plotting import plot_current
from cnsproject.plotting.plotting import plot_F_I

from typing import Tuple, Callable, Iterable, Union

class phase1():
    def __init__(
        self,
        shape: Iterable[int],
        time: int = 100,
        v_rest: Union[float, torch.Tensor] = -70.,
        threshold: Union[float, torch.Tensor] = -50.,
        tau: Union[float, torch.Tensor] = 10,
        dt: Union[float, torch.Tensor]= 1,
        R: Union[float, torch.Tensor] = 1.,
        current_interval: Tuple[int, int] = (0, 40),
        current_step: int = 10,
        current_threshold: int = 5,
        ** kwargs
    ) -> None:            
        """
        Phase 1 of the project. Includes the two main functions
        of part2 and part3.

        Parameters
        ----------
        shape : Iterable[int]
            Define the topology of neurons in the population.
        time : int, optional
            Simulation time steps.
        v_rest : Union[float, torch.Tensor], optional
            Resting membrane voltage. The default is -70..
        threshold : Union[float, torch.Tensor], optional
            Spike threshold voltage. The default is -50..
        tau : Union[float, torch.Tensor], optional
            Time constant of neuron voltage decay. The default is 10.
        dt : Union[float, torch.Tensor], optional
            Time resolution. The default is 1.
        R : Union[float, torch.Tensor], optional
            Resistance. The default is 1..
        current_interval : Tuple[int, int], optional
            Current interval to simulate the input current to neurons. The default is (0, 40).
        current_step : int, optional
            Steps value in current interval. The default is 10.
        current_threshold : int, optional
            Threshold value for threshold current function. The default is 5.

        Returns
        -------
        None

        """
        self.shape = shape
        self.time = time
        self.v_rest = v_rest
        self.threshold = threshold
        self.tau = tau
        self.dt = dt
        self.R = R
        self.current_interval = current_interval
        self.current_step = current_step
        self.current_threshold = current_threshold
        
    def current_function_maker(
        self,
        I: int = 10,
        constant: bool = True,
        I_interval: Tuple[int, int] = (0, 0),
    ) -> Callable[[int], int]:
        """
        Makes current function base on being constant or not
        
        """
        def current(t: int):
            if t > self.current_threshold:
                return I
            return 0
        
        x = (I_interval[0] - I_interval[1]) * torch.rand(self.time+1) + I_interval[1] + I
        def current_rand(t: int):
            return x[int(t)]
        
        if constant:
            return current
        return current_rand
    
    def Frequency_s(self, s: torch.Tensor) -> float:
        """
        Calculates spikes frequency.
        
        """
        count = 0
        for i in range(len(s)):
            if s[i] == True:
                count += 1
        return count/self.time
    
    def LIF_Simulation(self, current: Callable[[int], int]) -> Tuple[torch.Tensor, torch.Tensor]:
        neuron = LIFPopulation(shape=self.shape,
                               v_rest=self.v_rest,
                               threshold=self.threshold,
                               tau=self.tau,
                               dt=self.dt,
                               R=self.R)
        
        monitor = Monitor(neuron, state_variables=["s", "v"])
        monitor.set_time_steps(self.time, self.dt)
        monitor.reset_state_variables()
        for t in range(self.time):
            input_trace = torch.Tensor([current(t=t)])
            neuron.forward(input_trace)
            monitor.record()
        s = monitor.get("s")
        v = monitor.get("v")
        
        return s, v

    def part2(self) -> None:
        currents = torch.Tensor([])
        frequencies = torch.Tensor([])
        for I in range(self.current_interval[0], self.current_interval[1]+1 , self.current_step):
            currents = torch.cat((currents, torch.Tensor([I])))
            
            current = self.current_function_maker(I=I)
            
            s, v = self.LIF_Simulation(current=current)
            
            f = self.Frequency_s(s)
            frequencies = torch.cat((frequencies, torch.Tensor([f])))
            
            plot_voltage(v=v, s=s, time=(0, self.time), dt=self.dt, 
                         threshold=self.threshold, v_rest=self.v_rest)
            plot_current(current=current, time=(0, self.time), dt=self.dt)   
        plot_F_I(currents, frequencies)

    def part3(self, I: int = 20, noise: int = 5) -> None:
        current = self.current_function_maker(I=I, constant=False, I_interval=(-noise, noise))
        
        s, v = self.LIF_Simulation(current=current)   
        
        plot_voltage(v=v, s=s, time=(0, self.time), dt=self.dt, 
                     threshold=self.threshold, v_rest=self.v_rest)
        plot_current(current=current, time=(0, self.time), dt=self.dt) 

if __name__ == "__main__":
    p = phase1(shape=(1,),
               time=100,
               v_rest=-65,
               threshold=-50,
               tau=20,
               dt=1,
               R=1,
               current_interval=(0, 50),
               current_step=5,
               current_threshold=5)
    # Uncomment each to see the results
    # p.part2()
    p.part3(I=29, noise=5)