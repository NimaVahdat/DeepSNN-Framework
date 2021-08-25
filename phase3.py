import torch
from cnsproject.network.neural_populations import LIFPopulation, ELIFPopulation, AELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection 
from cnsproject.plotting.plotting import plot_current, raster, population_activity
from typing import Tuple, Callable, Iterable, Union

class phase3():
    def __init__(
        self,
        N: int,
        time: int = 100,
        dt: Union[float, torch.Tensor]= 1,
        ** kwargs
    ) -> None:            

        self.shape_exc = (int(N*0.8),)
        self.shape_inh = (int(N*0.2),)
        self.time = time
        self.dt = dt
        
        
    def current_maker(self, mean, std) -> Callable[[int], int]:
        torch.manual_seed(13)
        x = torch.empty(self.time + 1).normal_(mean=mean, std=std)
        y = torch.Tensor([0])
        x = torch.cat((y,x))
        def current_rand(t: int):
            return x[int(t)]
        return current_rand        
        
    def pop_maker(
            self,
            shape: Iterable[int],
            model = "LIF",
            is_inhibitory: bool = False,
            v_rest: Union[float, torch.Tensor] = -70.,
            threshold: Union[float, torch.Tensor] = -50.,
            tau: Union[float, torch.Tensor] = 15,
            dt: Union[float, torch.Tensor]= 1,
            R: Union[float, torch.Tensor] = 1.,
            delta_t: int = 1.,
            theta_rh: float = -55.,
            tau_w: Union[float, torch.Tensor] = 5,
            w: Union[float, torch.Tensor] = 2,
            a: Union[float, torch.Tensor] = 5,
            b: Union[float, torch.Tensor] = 2):
        if model == "LIF":
            neuron = LIFPopulation(shape=shape,
                                   is_inhibitory=is_inhibitory,
                                   v_rest=v_rest,
                                   threshold=threshold,
                                   tau=tau,
                                   dt=dt,
                                   R=R)
        elif model == "ELIF":
            neuron = LIFPopulation(shape=shape,
                                   is_inhibitory=is_inhibitory,
                                   v_rest=v_rest,
                                   threshold=threshold,
                                   tau=tau,
                                   dt=dt,
                                   R=R,
                                   theta_rh=theta_rh,
                                   delta_t=delta_t)
        elif model == "AELIF":
            neuron = AELIFPopulation(shape=shape,
                                   is_inhibitory=is_inhibitory,
                                   v_rest=v_rest,
                                   threshold=threshold,
                                   tau=tau,
                                   dt=dt,
                                   R=R,
                                   theta_rh=theta_rh,
                                   delta_t=delta_t,
                                   tau_w=tau_w,
                                   w=w,
                                   a=a,
                                   b=b)
        
        return neuron
    
    def Simulation(self, pop_exc, pop_inh, mean, std):
        current = self.current_maker(mean, std)
        monitor_exc = Monitor(pop_exc, state_variables=["s", "v"])
        monitor_exc.set_time_steps(self.time, self.dt)
        monitor_exc.reset_state_variables()
        
        monitor_inh = Monitor(pop_inh, state_variables=["s", "v"])
        monitor_inh.set_time_steps(self.time, self.dt)
        monitor_inh.reset_state_variables()        
        
        connect_exc_to_inh = RandomConnection(pop_exc, pop_inh, C=20,control=1) ####
        connect_inh_to_exc = RandomConnection(pop_inh, pop_exc, C=10,control=1) ####
        connect_inside_exc = DenseConnection(pop_exc, pop_exc, inside=True,control=1)

        for t in range(self.time):
            input_I = torch.Tensor([current(t=t)])
            
            traces_inh = connect_exc_to_inh.compute(pop_exc.s)
            traces_exc = connect_inh_to_exc.compute(pop_inh.s) +\
                connect_inside_exc.compute(pop_exc.s)

            pop_exc.forward(I=input_I, traces=traces_exc)
            pop_inh.forward(I=input_I, traces=traces_inh)
            
            monitor_exc.record()
            monitor_inh.record()
            
        s_exc = monitor_exc.get("s")
        s_inh = monitor_inh.get("s")
        
        population_activity(s_exc, "of exc")
        population_activity(s_inh, "of inh") 

        plot_current(current=current, time=(0, self.time), dt=self.dt)           
        
        raster(s_exc, s_inh)
        
if __name__ == "__main__":
    p = phase3(100)
    pop_exc=p.pop_maker(p.shape_exc, tau=20)
    pop_inh=p.pop_maker(p.shape_inh, tau=10, is_inhibitory=True)
    
    p.Simulation(pop_exc, pop_inh, 48, 15)