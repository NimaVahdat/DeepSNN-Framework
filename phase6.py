import torch
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.learning.learning_rules import STDP, FlatSTDP
from cnsproject.network.connections import DenseConnection, RandomConnection 
from cnsproject.plotting.plotting import plot_w, raster, population_activity, plot_voltage

from cnsproject.network.monitors import Monitor

import random

x1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0])
x2 = x1.flip(0)

time = 1400


# =============================================================================
# STDP     Uncomment
# =============================================================================
learning_rule = STDP
input_pop = InputPopulation((10,), tau_s=10)
out_pop = LIFPopulation((2,), threshold=-55, tau=10, v_rest=-65, tau_s=10, R=5)
connect = DenseConnection(input_pop, out_pop,  [0.1, 0.1], learning_rule=learning_rule, dt=1.0)


# # =============================================================================
# # FlatSTDP
# # =============================================================================
# learning_rule = FlatSTDP
# input_pop = InputPopulation((10,), tau_s=10, additive_spike_trace=False)
# out_pop = LIFPopulation((2,), threshold=-60, tau=50, v_rest=-65, tau_s=10, R=40, additive_spike_trace=False)
# connect = DenseConnection(input_pop, out_pop,  [0.1, 0.1], learning_rule=learning_rule, dt=1.0)


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

j=0
x=[x1,x2]
for i in range(time):
    j += 1
    if j==100:
        j=0
        x = x[::-1]
    y = x[0]

    y = torch.poisson(y) > 0
    # print(y)
    input_pop.forward(y)
    t = connect.compute(input_pop.s)
    out_pop.forward(I=0.7, traces=t)
    # print(out_pop.s)
    connect.update()

    monitor_in.record()
    monitor_out.record()
    monitor_w.record()
    
out_s = monitor_out.get("s")
out_v = monitor_out.get("v")
in_s = monitor_in.get("s")
w = monitor_w.get("w")

raster(in_s, out_s)

plot_w(w)
