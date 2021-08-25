import torch
from cnsproject.network.neural_populations import LIFPopulation, ELIFPopulation, AELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection 
from cnsproject.plotting.plotting import plot_current, raster, population_activity
from typing import Tuple, Callable, Iterable, Union

class phase4():
    def __init__(
        self,
        N: int,
        time: int = 100,
        dt: Union[float, torch.Tensor]= 1,
        ** kwargs
    ) -> None:            

        self.shape_exc = (int(N*0.8/2),)
        self.shape_inh = (int(N*0.2),)
        self.time = time
        self.dt = dt
        
        
    def current_maker(self, mean, std, threshold=None, jump=60) -> Callable[[int], int]:
        torch.manual_seed(16)
        if threshold != None:
            a = torch.empty(threshold + 1).normal_(mean=mean+jump, std=std)
            b = torch.empty(self.time - threshold).normal_(mean=mean, std=std)
            x = torch.cat((b,a))
        else:
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
    
    def Simulation(self, pop_inh, pop_exc1, pop_exc2, mean: list, std: list):
        current0 = self.current_maker(mean[0], std[0])
        current1 = self.current_maker(mean[1], std[1], threshold=50)
        current2 = self.current_maker(mean[2], std[2])
        
        monitor_exc1 = Monitor(pop_exc1, state_variables=["s", "v"])
        monitor_exc1.set_time_steps(self.time, self.dt)
        monitor_exc1.reset_state_variables()
        
        monitor_exc2 = Monitor(pop_exc2, state_variables=["s", "v"])
        monitor_exc2.set_time_steps(self.time, self.dt)
        monitor_exc2.reset_state_variables()
        
        monitor_inh = Monitor(pop_inh, state_variables=["s", "v"])
        monitor_inh.set_time_steps(self.time, self.dt)
        monitor_inh.reset_state_variables()        
        
        connect_exc1_to_inh = DenseConnection(pop_exc1, pop_inh, C=25,control=1)
        connect_inh_to_exc1 = DenseConnection(pop_inh, pop_exc1, C=13,control=1)

        connect_exc2_to_inh = DenseConnection(pop_exc2, pop_inh, C=25,control=1)
        connect_inh_to_exc2 = DenseConnection(pop_inh, pop_exc2, C=13,control=1)        
        
        connect_exc1_to_exc2 = DenseConnection(pop_exc1, pop_exc2, C=5,control=1)
        connect_exc2_to_exc1 = DenseConnection(pop_exc2, pop_exc1, C=5,control=1)
        
        connect_inside_exc1 = RandomConnection(pop_exc1, pop_exc1, insode=True, C=10,control=1)
        connect_inside_exc2 = RandomConnection(pop_exc2, pop_exc2, inside=True, C=10,control=1)
        connect_inside_inh = RandomConnection(pop_inh, pop_inh, inside=True, C=10,control=1)
        
        for t in range(self.time):
            input_I_inh = torch.Tensor([current0(t=t)] * pop_inh.shape[0])
            input_I_exc1 = torch.Tensor([current1(t=t)] * pop_exc1.shape[0])
            input_I_exc2 = torch.Tensor([current2(t=t)] * pop_exc2.shape[0])
            noise_inh = torch.empty(input_I_inh.size(0)).normal_(0, 5)
            noise_exc1 = torch.empty(input_I_exc1.size(0)).normal_(0, 5)
            noise_exc2 = torch.empty(input_I_exc2.size(0)).normal_(0, 5)
            
            traces_inh = connect_exc1_to_inh.compute(pop_exc1.s) +\
                connect_exc2_to_inh.compute(pop_exc2.s) + \
                    connect_inside_inh.compute(pop_inh.s)
                
            traces_exc1 = connect_inh_to_exc1.compute(pop_inh.s) +\
                connect_exc2_to_exc1.compute(pop_exc2.s) + \
                    connect_inside_exc1.compute(pop_exc1.s)

            traces_exc2 = connect_inh_to_exc2.compute(pop_inh.s) +\
                connect_exc1_to_exc2.compute(pop_exc1.s) + \
                    connect_inside_exc2.compute(pop_exc2.s)
                    
            pop_inh.forward(I=input_I_inh-noise_inh, traces=traces_inh)
            pop_exc1.forward(I=input_I_exc1-noise_exc1, traces=traces_exc1)
            pop_exc2.forward(I=input_I_exc2-noise_exc2, traces=traces_exc2)
            
            monitor_inh.record()
            monitor_exc1.record()
            monitor_exc2.record()
            
        s_inh = monitor_inh.get("s")
        s_exc1 = monitor_exc1.get("s") 
        s_exc2 = monitor_exc2.get("s")
        
        population_activity(s_inh, "of inh") 
        population_activity(s_exc1, "of exc1")
        population_activity(s_exc2, "of exc2")        
        
        plot_current(current=current0, time=(0, self.time), dt=self.dt, label="inh input current")           
        plot_current(current=current1, time=(0, self.time), dt=self.dt, label="exc1 input current") 
        plot_current(current=current2, time=(0, self.time), dt=self.dt, label="exc2 input current") 
        
        
        raster(s_exc1, label="exc1")
        raster(s_exc2, label="exc2")
        raster(s_inh=s_inh, label="inh")
        
if __name__ == "__main__":
    p = phase4(100)
    pop_exc1=p.pop_maker(p.shape_exc, tau=20)
    pop_exc2=p.pop_maker(p.shape_exc, tau=20)
    pop_inh=p.pop_maker(p.shape_inh, tau=10, R=1, is_inhibitory=True)
    
    p.Simulation(pop_inh, pop_exc1, pop_exc2, [20, 45, 45], [5, 5, 5])