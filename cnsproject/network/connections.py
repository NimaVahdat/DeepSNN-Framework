"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence

import torch

from .neural_populations import NeuralPopulation
from ..learning.learning_rules import LearningRule, NoOp

from torch.nn import Module, Parameter
import numpy as np

import torch.nn.functional as F

class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()

        # assert isinstance(pre, NeuralPopulation), \
        #     "Pre is not a NeuralPopulation instance"
        # assert isinstance(post, NeuralPopulation), \
        #     "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post

        self.weight_decay = weight_decay

        # from ..learning.learning_rules import NoOp

        learning_rule = kwargs.get('learning_rule', NoOp)

        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.wmin = kwargs.get('wmin', 0)
        self.wmax = kwargs.get('wmax', 1)
        self.norm = kwargs.get('norm', None)

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", True)

        if learning:
            self.learning_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.pre = pre
        self.post = post
        self.lr = lr
        self.weight_decay = weight_decay
        
        inside = kwargs.get("inside", False)
        
        w = kwargs.get("w", None)
        # mean = kwargs.get("mean", 18)
        # std = kwargs.get("std", 1.8)
        control = kwargs.get("control", 1)
        if w is None:
            # w = control*torch.empty(*pre.shape, *post.shape).normal_(mean=1, std=0.5)
            # w = w * 0.1 + 0.45
            w = control * torch.rand(*pre.shape, *post.shape)#clamp(torch.rand(*pre.shape, *post.shape), self.wmin, self.wmax)
        if inside:
            torch.diagonal(w, 0).zero_()
        # self.w = Parameter(w, requires_grad=False)
        self.w = w
        self.learning_rule = kwargs.get("learning_rule", NoOp)
        tau_c = kwargs.get("tau_c", 100)
        dt = kwargs.get("dt", 1.)
        self.rule = self.learning_rule(self, lr, weight_decay, tau_c=tau_c , dt=dt)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        post = s.float() @ self.w
        
        if self.pre.is_inhibitory == True:
            post *= -1 
        
        return post

    def update(self, **kwargs) -> None:
        d = kwargs.get("d", None)
        self.rule.update(d=d)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.pre = pre
        self.post = post
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        w = kwargs.get("w", None)
        mean = kwargs.get("mean", 0.4)
        std = kwargs.get("std", 0.2)
        control = kwargs.get("control", 1)
        if w is None:
            # w = control*torch.empty(*pre.shape, *post.shape).normal_(mean=mean, std=std)
            w = control * torch.rand(*pre.shape, *post.shape)
        inside = kwargs.get("inside", False)
        self.C = kwargs.get("C", 1)
        
        rand_mat = torch.rand(*post.shape, *pre.shape)
        if inside:
            torch.diagonal(rand_mat, 0).zero_()
            
        k_th_quant = torch.topk(rand_mat, self.C)[0][:, -1:]
        mask = rand_mat >= k_th_quant
        if len(mask.size()) > 1:
            self.mask = torch.t(mask)
        else:
            self.mask = mask.view(mask.size(0), -1)
            
        w = w * self.mask

        self.w = Parameter(w, requires_grad=False)
        
        
        self.learning_rule = kwargs.get("learning_rule", NoOp)
        self.rule = self.learning_rule(self, lr, weight_decay)
        
    def compute(self, s: torch.Tensor) -> torch.Tensor:
        post = s.float() @ self.w
        
        if self.pre.is_inhibitory == True:
            post *= -1 
        
        return post

    def update(self, **kwargs) -> None:
        d = kwargs.get("d", None)
        self.rule.update(d=d)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()


class ConvolutionalConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        filter_size: Union[int, int] = [3, 3],
        stride: int = 2,
        padding: int = 0,
        feature_n: int = 1,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.filter_size = filter_size
        self.stride = stride
        h= filter_size[0]
        self.padding = (h-1) // 2 if h%2==1 else h//2
        self.feature_n = feature_n
        
        self.input_feature, input_height, input_width = 1, pre.shape[0], pre.shape[1]
        self.out_feature = post.shape[1] ############################
        
        self.w = kwargs.get("w", None)
        if self.w == None:
            self.w = torch.clamp(torch.rand(self.out_feature, self.input_feature, *self.filter_size), self.wmin, self.wmax)
        
        # self.w = kwargs.get("w", None)
        # if self.w == None:
        #     self.w = torch.rand(feature_n, *filter_size)
            
        self.learning_rule = kwargs.get("learning_rule", NoOp)
        self.rule = self.learning_rule(self, lr, weight_decay) 
        
        
    def compute(self, s: torch.Tensor) -> None:
        mask = torch.ones(1, 1, *s.shape) * s
        bias = torch.zeros(self.out_feature)
        # print(F.conv2d(mask.float(), self.w, bias, stride=self.stride, padding=self.padding).shape)
        return F.conv2d(mask.float(), self.w, bias, stride=self.stride, padding=self.padding)


    def update(self, **kwargs) -> None:
        control = kwargs.get("control", 1)
        self.rule.conv2d_update(control = control)
        

    def reset_state_variables(self) -> None:
        super().reset_state_variables()




class PoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        pooling_size: Union[int, int] = [3, 3],
        stride: int = 1,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.pooling_size = pooling_size
        self.stride = stride

    def compute(self, s: torch.Tensor) -> None:
        feature = s.shape[0]
        h, w = s[0].shape[0]//self.stride, s[0].shape[1]//self.stride
        results = torch.zeros(feature, h, w)
        for k in range(len(s)):
            x = 0
            for i in range(h):
                y = 0
                for j in range(w):
                    results[k][i][j] = torch.max(s[k, x:x+h, y:y+w])
                    y += self.stride
                x += self.stride
        return results

    def update(self, **kwargs) -> None:

        pass

    def reset_state_variables(self) -> None:

        pass
