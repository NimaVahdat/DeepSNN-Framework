"""
Module for neuronal dynamics and populations.
"""

from functools import reduce
from abc import abstractmethod
from operator import mul
from typing import Union, Iterable

import torch


class NeuralPopulation(torch.nn.Module):
    """
    Base class for implementing neural populations.

    The most important attribute of each neural population is its `shape` which indicates the\
    number and/or architecture of the neurons in it. When there are connected populations, each\
    pre-synaptic population will have an impact on the post-synaptic one in case of spike. This\
    spike might be persistent for some duration of time and with some decaying magnitude. To\
    handle this coincidence, four attributes are defined:
    - `spike_trace` is a boolean indicating whether to record the spike trace in each time step.
    - `additive_spike_trace` would indicate whether to save the accumulated traces up to the\
        current time step.
    - `tau_s` will show the duration by which the spike trace persists by a decaying manner.
    - `trace_scale` is responsible for the scale of each spike at the following time steps.\
        Its value is only considered if `additive_spike_trace` is set to `True`.

    Make sure to call `reset_state_variables` before starting the simulation to allocate\
    and/or reset the state variables such as `s` (spikes tensor) and `traces` (trace of spikes).\
    Also do not forget to set the time resolution (dt) for the simulation.

    Each simulation step is defined in `forward` method. You can use the utility methods (i.e.\
    `compute_potential`, `compute_spike`, `refractory_and_reset`, and `compute_decay`) to break\
    the differential equations into smaller code blocks and call them within `forward`. Make\
    sure to call methods `forward` and `compute_decay` of `NeuralPopulation` in child class\
    methods; As it provides the computation of spike traces (not necessary if you are not\
    considering the traces). The `forward` method can either work with current or spike trace.\
    You can easily work with any of them you wish. When there are connected populations, you\
    might need to consider how to convert the pre-synaptic spikes into current or how to\
    change the `forward` block to support spike traces as input.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    is_inhibitory : False, Optional
        Whether the neurons are inhibitory or excitatory. The default is False.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 15.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.shape = shape
        self.n = reduce(mul, self.shape)
        self.spike_trace = spike_trace
        self.additive_spike_trace = additive_spike_trace

        if self.spike_trace:
            self.register_buffer("traces", torch.zeros(*self.shape))
            self.register_buffer("tau_s", torch.tensor(tau_s))

            if self.additive_spike_trace:
                self.register_buffer("trace_scale", torch.tensor(trace_scale))

            self.register_buffer("trace_decay", torch.empty_like(self.tau_s))

        self.is_inhibitory = is_inhibitory
        self.learning = learning

        self.register_buffer("s", torch.zeros(*self.shape, dtype=torch.bool))
        self.dt = 1.0

    @abstractmethod
    def forward(self, traces: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        traces : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        if self.spike_trace:
            self.traces *= self.trace_decay

            if self.additive_spike_trace:
                # print(self.trace_decay, self.trace_scale)
                self.traces += self.trace_scale * self.s.float()
            else:
                self.traces.masked_fill_(self.s, 1)

    @abstractmethod
    def compute_potential(self) -> None:
        """
        Compute the potential of neurons in the population.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_spike(self) -> None:
        """
        Compute the spike tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Refractor and reset the neurons.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Set the decays.

        Returns
        -------
        None

        """
        # self.dt = torch.tensor(self.dt)

        if self.spike_trace:
            self.trace_decay = torch.exp(-self.dt/self.tau_s)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        self.s.zero_()

        if self.spike_trace:
            self.traces.zero_()

    def train(self, mode: bool = True) -> "NeuralPopulation":
        """
        Set the population's training mode.

        Parameters
        ----------
        mode : bool, optional
            Mode of training. `True` turns on the training while `False` turns\
            it off. The default is True.

        Returns
        -------
        NeuralPopulation

        """
        self.learning = mode
        return super().train(mode)


class InputPopulation(NeuralPopulation):
    """
    Neural population for user-defined spike pattern.

    This class is implemented for future usage. Extend it if needed.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            learning=learning,
        )

    def forward(self, traces: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        traces : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        self.s = traces
        self.compute_decay()
        super().forward(traces)

    def compute_decay(self) -> None:
        super().compute_decay()

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()


class LIFPopulation(NeuralPopulation):
    """
    Layer of Leaky Integrate and Fire neurons.

    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        v_rest: float = -70.,
        threshold: float = -50.,
        tau: float = 10,
        dt: float= 1,
        R: float = 10.,
        **kwargs
    ) -> None:
        """
        Instantiates a layer of LIF neurons.

        Parameters
        ----------
        shape : Iterable[int]
            Define the topology of neurons in the population.
        spike_trace : bool, optional
            Specify whether to record spike traces. The default is True.
        additive_spike_trace : bool, optional
            Specify whether to record spike traces additively. The default is True.
        tau_s : Union[float, torch.Tensor], optional
            Time constant of spike trace decay. The default is 10..
        trace_scale : Union[float, torch.Tensor], optional
            The scaling factor of spike traces. The default is 1..
        is_inhibitory : bool, optional
            Whether the neurons are inhibitory or excitatory. The default is False.
        learning : bool, optional
            Define the training mode. The default is True.
        v_rest : Union[float, torch.Tensor], optional
            Resting membrane voltage. The default is -70..
        threshold : Union[float, torch.Tensor], optional
            Spike threshold voltage. The default is -50..
        tau : Union[float, torch.Tensor], optional
            Time constant of neuron voltage decay. The default is 10.
        dt : Union[float, torch.Tensor], optional
            Time resolution. The default is 1.
        R : Union[float, torch.Tensor], optional
            Resistance. The default is 5..

        Returns
        -------
        None

        """
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        self.shape = shape
        self.v_rest = torch.full(shape, v_rest)
        self.v = torch.full(shape, v_rest)
        self.threshold = torch.full(shape, threshold)
        self.R = torch.full(shape, R)
        self.dt = torch.full(shape, dt)
        self.tau = torch.full(shape, tau)
        
        self.register_buffer(
            "decay", torch.zeros(*self.shape)
        )  # Set in compute_decays.
        
        self.is_inhibitory = is_inhibitory
        
    def forward(self, traces: torch.Tensor, I: torch.Tensor = None) -> None:
        """
        Runs a single simulation step.
        
        :param traces: Inputs to the layer.
        """
        
        self.I = I
        if self.I == None:
            self.I = torch.zeros(*self.shape)
        
        # Neuron voltage decay (per timestep).
        self.compute_decay()
        
        self.compute_potential()
        
        # print(self.v.shape, traces.shape)
        self.v += traces
        
        # Check for spiking neurons.
        self.compute_spike()
        # Refractoriness and voltage reset
        # self.refractory_and_reset()
        
        super().forward(traces)

    def compute_potential(self) -> None:
        """
        
        Neural dynamics for computing 
        the potential of LIF neurons.
        """
        # print(self.v.shape, self.I.shape)
        tau_du_dt = -(self.v - self.v_rest) + (self.R * self.I)
        self.v = self.v + (tau_du_dt * (self.dt / self.tau))

    def compute_spike(self) -> None:
        """
        Check for spiking neuron.
        """
        self.s = self.v >= self.threshold

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Resets relevant state variables.
        """
        # print(self.s.shape, self.v_rest.shape, "here")
        self.v.masked_fill_(self.s, -70.)


    @abstractmethod
    def compute_decay(self) -> None:
        """
        Sets the relevant decays.
        """
        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tau)
        
        # Decay voltage.
        self.v = self.decay * (self.v - self.v_rest) + self.v_rest

class ELIFPopulation(NeuralPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.

    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        v_rest: float = -70.,
        threshold: float = -50.,
        tau: float = 10,
        dt: float= 1,
        R: float = 5.,
        delta_t: int = 1.,
        theta_rh: float = -55.,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )


        self.shape = shape
        self.v_rest = torch.full(shape, v_rest)
        self.v = torch.full(shape, v_rest)
        self.threshold = torch.full(shape, threshold)
        self.R = torch.full(shape, R)
        self.dt = torch.full(shape, dt)
        self.tau = torch.full(shape, tau)

        self.delta_t = torch.full(shape, delta_t)
        self.theta_rh = torch.full(shape, theta_rh)
        
        self.is_inhibitory = is_inhibitory

    def forward(self, traces: torch.Tensor, I: torch.Tensor) -> None:
        """
        Runs a single simulation step.
        
        :param traces: Inputs to the layer.
        """
        self.reset_state_variables()
        
        self.I = I
        
        # Neuron voltage decay (per timestep).
        self.compute_decay()
        
        self.compute_potential()
        
        self.v += traces
        
        # Check for spiking neurons.
        self.compute_spike()
        # Refractoriness and voltage reset
        self.refractory_and_reset()
        
        super().forward(traces)        

    def compute_potential(self) -> None:
        tau_du_dt = -(self.v - self.v_rest) + self.delta_t *\
            torch.exp((self.v - self.theta_rh)/self.delta_t) + self.R * self.I
        
        du = tau_du_dt * self.dt / self.tau
        self.v = self.v + du

    def compute_spike(self) -> None:
        
        self.s = self.v >= self.threshold

    @abstractmethod
    def refractory_and_reset(self) -> None:
        
        self.v.masked_fill_(self.s, self.v_rest[0])

    @abstractmethod
    def compute_decay(self) -> None:
        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tau_s)
        
        # Decay voltage.
        self.v = self.decay * (self.v - self.v_rest) + self.v_rest

class AELIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Exponential Leaky Integrate and Fire neurons.
    
    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        v_rest: float = -70.,
        threshold: float = -50.,
        tau: float = 10,
        dt: float= 1,
        R: float = 5.,
        delta_t: int = 1.,
        theta_rh: float = -55.,
        tau_w: float = 5,
        w: float = 2,
        a: float = 7,
        b: float = 4,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )


        self.shape = shape
        self.v_rest = torch.full(shape, v_rest)
        self.v = torch.full(shape, v_rest)
        self.threshold = torch.full(shape, threshold)
        self.R = torch.full(shape, R)
        self.dt = torch.full(shape, dt)
        self.tau = torch.full(shape, tau)

        self.delta_t = torch.full(shape, delta_t)
        self.theta_rh = torch.full(shape, theta_rh)
        
        self.tau_w = torch.full(shape, tau_w)
        self.w = torch.full(shape, w)
        self.a = torch.full(shape, a)
        self.b = torch.full(shape, b)
        
        self.is_inhibitory = is_inhibitory

    def forward(self, traces: torch.Tensor, I: torch.Tensor) -> None:

        self.reset_state_variables()
        
        self.I = I
        
        tau_dw_dt = self.a * (self.v - self.v_rest) - self.w +\
            self.b * self.tau_w * self.s
        dw = tau_dw_dt * self.dt / self.tau_w
        self.w = self.w + dw 
        
        # Neuron voltage decay (per timestep).
        self.compute_decay()
           
    
        self.compute_potential()
        
        self.v += traces
        
        # Check for spiking neurons.
        self.compute_spike()

        tau_dw_dt = self.a * (self.v - self.v_rest) - self.w +\
            self.b * self.tau_w * self.s
        dw = tau_dw_dt * self.dt / self.tau_w
        self.w = self.w + dw 
        
        # Refractoriness and voltage reset
        self.refractory_and_reset()
        
        super().forward(traces)

    def compute_potential(self) -> None:

        tau_du_dt = -(self.v - self.v_rest) + self.delta_t *\
            torch.exp((self.v - self.theta_rh)/self.delta_t) - self.R * self.w + self.R * self.I
        
        du = tau_du_dt * self.dt / self.tau
        self.v = self.v + du
            
    def compute_spike(self) -> None:
        self.s = self.v >= self.threshold

    @abstractmethod
    def refractory_and_reset(self) -> None:

        self.v.masked_fill_(self.s, self.v_rest[0])

    @abstractmethod
    def compute_decay(self) -> None:

        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tau_s)

        # Decay voltage.
        self.v = self.decay * (self.v - self.v_rest) + self.v_rest