import torch
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.learning.learning_rules import RSTDP, FlatRSTDP
from cnsproject.learning.rewards import Reward
from cnsproject.network.connections import DenseConnection, RandomConnection 
from cnsproject.plotting.plotting import plot_w, raster, population_activity, plot_voltage, plot_reward

from cnsproject.network.monitors import Monitor

import random


def DA(pattern_in, pattern_out):
    # print(pattern_in, pattern_out)
    
    if pattern_out[0] == False and pattern_out[1] == False:
        # print(0)
        return 0
    elif pattern_out[0] == True and pattern_out[1] == True:
        return -1
    elif pattern_in == False and pattern_out[0] == True:
        # print(0.1)
        return 1
    elif pattern_in == True and pattern_out[1] == True:
        # print(0.1)
        return 1
    # print(-0.1)
    return -1


for i in range(40):
    x1 = torch.tensor([.7, .7, .7, .7, .7, 0, 0, 0, 0, 0])
    x2 = x1.flip(0)
    
    time = 1600
    
    input_pop = InputPopulation((10,), tau_s=10)
    out_pop = LIFPopulation((2,), threshold=-55, tau=10, v_rest=-65, R=1, tau_s=10)
    
    connect = DenseConnection(input_pop, out_pop,  [0.025, 0.025], learning_rule=RSTDP, tau_c=500, dt=1.0)
    
    reward = Reward(DA, tau_d=10, dt=1)
    
    input_pop.reset_state_variables()
    out_pop.reset_state_variables()
    
    
    monitor_out = Monitor(out_pop, state_variables=["s","v"])
    monitor_out.set_time_steps(time, 1)
    monitor_out.reset_state_variables()
    
    monitor_w = Monitor(connect, state_variables=["w"])
    monitor_w.set_time_steps(time, 1)
    monitor_w.reset_state_variables()
    
    monitor_in = Monitor(input_pop, state_variables=["s"])
    monitor_in.set_time_steps(time, 1)
    monitor_in.reset_state_variables()
    
    # monitor_reward = Monitor(reward, state_variables=["d"])
    # monitor_reward.set_time_steps(time, 1)
    # monitor_reward.reset_state_variables()
    
            
    dd = []
    j=0
    x=[x1,x2]
    flag = False
    for i in range(time):
        j += 1
        if j==100:
            j=0
            x = x[::-1]
            if flag == True:
                flag = False
            else:
                flag = True
        y = x[0]
    
        y = torch.poisson(y) > 0
        # print(y)
        input_pop.forward(y)
        t = connect.compute(input_pop.s)
        out_pop.forward(I=0.5, traces=t)
        # print(out_pop.s)
        
        
        d = reward.compute(flag, out_pop.s)
        connect.update(d=d)
    
        monitor_in.record()
        monitor_out.record()
        monitor_w.record()
        dd.append(reward.d)
        
    out_s = monitor_out.get("s")
    out_v = monitor_out.get("v")
    in_s = monitor_in.get("s")
    w = monitor_w.get("w")
    
    raster(in_s, out_s)
    
    plot_w(w)
    
    # d = monitor_reward.get("d")
    plot_reward(dd)
    # print(dd)