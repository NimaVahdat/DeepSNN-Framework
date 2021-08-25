"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# from ..network.connections import AbstractConnection
AbstractConnection = Union

class LearningRule(ABC):
    """
    Abstract class for defining learning rules.
    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.
    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.
    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.lr = torch.tensor(lr, dtype=torch.float)

        self.weight_decay = 1 - weight_decay if weight_decay else 1.

    def update(self) -> None:
        """
        Abstract method for a learning rule update.
        Returns
        -------
        None
        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        if (
            self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.w.clamp_(self.connection.wmin,
                                     self.connection.wmax)


class NoOp(LearningRule):
    """
    Learning rule with no effect.
    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.
        Returns
        -------
        None
        """
        super().update()


class STDP(LearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.
    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.
        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.connection = connection
        self.lr = lr
        self.weight_decay
        
        self.dt = kwargs.get("dt", 1.)

    def update(self, **kwargs) -> None:
        """
        TODO.
        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        mask = torch.ones(*self.connection.w.size())
        
        pre_s = self.connection.pre.s

        if len(pre_s.size()) > 1:
            pre_s = torch.t(pre_s)
        else:
            pre_s = pre_s.view(pre_s.size(0), -1)
        pre_s = pre_s * mask
        
        post_traces = self.connection.post.traces * mask
        # print(post_traces)
        p1 = self.lr[0] * pre_s * post_traces
        
        
        post_s = self.connection.post.s * mask
        
        pre_traces = self.connection.pre.traces
        if len(pre_traces.size()) > 1:
            pre_traces = torch.t(pre_traces)
        else:
            pre_traces = pre_traces.view(pre_traces.size(0), -1)
        pre_traces = pre_traces * mask
        # print(pre_traces)
        p2 = self.lr[1] * post_s * pre_traces
        # print(self.lr[1])
        self.connection.w += ((-p1) + p2) * self.dt
        self.connection.w = torch.clamp(self.connection.w, self.connection.wmin, self.connection.wmax)

        super().update()
        
    def conv2d_update(self, control, **kwargs) -> None:
        out_feature, _, filter_height, filter_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        
        mask = torch.ones(1, 1, *self.connection.pre.s.shape)
        
        pre_traces = F.unfold(self.connection.pre.traces * mask, (filter_height, filter_width), padding=padding, stride=stride)
        pre_s = F.unfold(self.connection.pre.s.float() * mask, (filter_height, filter_width), padding=padding, stride=stride)
        
        post_traces = (self.connection.post.traces * control).view(1, out_feature, -1)
        post_s = (self.connection.post.s * control).view(1, out_feature, -1).float()

        A1 = torch.bmm(post_traces, pre_s.permute((0, 2, 1))).view(out_feature, 1, filter_height, filter_width)

        A2 = torch.bmm(post_s, pre_traces.permute((0, 2, 1))).view(out_feature, 1, filter_height, filter_width)
        print((-self.lr[0] * A1 + self.lr[1] * A2).sum())
        # print(A1.sum(), A2.sum())
        self.connection.w += (-1 * (self.lr[0] * A1)) + self.lr[1] * A2
        
        self.connection.w *= 50/self.connection.w.sum()
        # self.connection.w = torch.clamp(self.connection.w, self.connection.wmin, self.connection.wmax)

class FlatSTDP(LearningRule):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.
    Implement the dynamics of Flat-STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.
        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.connection = connection
        self.lr = lr
        self.weight_decay
        
        self.dt = kwargs.get("dt", 1.)

    def update(self, **kwargs) -> None:
        """
        TODO.
        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        mask = torch.ones(*self.connection.w.size())
        
        pre_s = self.connection.pre.s

        if len(pre_s.size()) > 1:
            pre_s = torch.t(pre_s)
        else:
            pre_s = pre_s.view(pre_s.size(0), -1)
        pre_s = pre_s * mask
        
        post_traces = self.connection.post.traces * mask

        p1 = self.lr[0] * pre_s * post_traces
        
        
        post_s = self.connection.post.s * mask
        
        pre_traces = self.connection.pre.traces
        if len(pre_traces.size()) > 1:
            pre_traces = torch.t(pre_traces)
        else:
            pre_traces = pre_traces.view(pre_traces.size(0), -1)
        pre_traces = pre_traces * mask

        p2 = self.lr[1] * post_s * pre_traces

        self.connection.w += ((-p1) + p2) * self.dt
        self.connection.w = torch.clamp(self.connection.w, self.connection.wmin, self.connection.wmax)

        super().update()

class RSTDP(LearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.
    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.
        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.connection = connection
        self.lr = lr
        self.weight_decay
        
        self.tau_c = kwargs.get("tau_c", 100)
        self.dt = kwargs.get("dt", 1)
        self.c = torch.zeros(*self.connection.pre.shape, *self.connection.post.shape)

    def update(self, **kwargs) -> None:
        """
        TODO.
        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        d = kwargs.get("d", None)
        
        mask = torch.ones(*self.connection.w.size())
        pre_s = self.connection.pre.s

        if len(pre_s.size()) > 1:
            pre_s = torch.t(pre_s)
        else:
            pre_s = pre_s.view(pre_s.size(0), -1)
        pre_s = pre_s * mask
        post_traces = self.connection.post.traces * mask
        p1 = self.lr[0] * pre_s * post_traces
        
        post_s = self.connection.post.s * mask
        
        pre_traces = self.connection.pre.traces
        if len(pre_traces.size()) > 1:
            pre_traces = torch.t(pre_traces)
        else:
            pre_traces = pre_traces.view(pre_traces.size(0), -1)
        pre_traces = pre_traces * mask
        p2 = self.lr[1] * post_s * pre_traces
        stdp = -p1 + p2  
        # print(stdp, ((pre_s + post_s) > 0))
        
        dc_dt = -self.c/self.tau_c + stdp * ((pre_s + post_s) > 0)
        self.c += dc_dt * self.dt
        # print(self.c)
        self.connection.w += self.c * d * self.dt
        self.connection.w = torch.clamp(self.connection.w, self.connection.wmin, self.connection.wmax)
        super().update()


class FlatRSTDP(LearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.
    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.
        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.
        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass