import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import *


import torch.nn.functional as F

class AbstractFilter(ABC):
    def __init__(
            self,
            size: int = 3,
            center_type: str = "on",
            stride: int = 1,
            **kwargs
    ) -> None:
        super().__init__()
        
        self.size = size
        self.stride = stride
        self.control = 1 if center_type == "on" else -1
        
        self.filter = self.filter_maker()
        
    @abstractmethod
    def filter_maker(self, **kwargs) -> None:
        """
        Creates filter
        
        Returns
        -------
        None
            
        """
        pass
    
    @abstractmethod
    def save_filterPic(self, name: str) -> None:
        """
        Saves filter picture

        Parameters
        ----------
        name : str

        Returns
        -------
        None

        """
        f = self.filter - self.filter.min()
        f = f/ f.max() * 255
        f = np.array(f)
        # if self.control == -1:
        #     f = 255 - f
        f = cv2.resize(f, (100, 100))
        cv2.imwrite(name + ".jpg", f)
    
    @abstractmethod
    def save_outputPic(self, name: str) -> None:
        """
        Saves output picture

        Parameters
        ----------
        name : str
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        y = self.result - self.result.min()
        y = y / y.max() * 255
        y = np.array(y)
        # if self.control == -1:
        #     y = 255 - y
        cv2.imwrite(name + ".jpg", y)
    
    @abstractmethod
    def __str__(self) -> torch.Tensor:
        """
        Prints genrated filter matrix

        Returns
        -------
        TYPE
            Filter.

        """
        
        return str(self.filter)
    
    @abstractmethod
    def __call__(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convolves filter on image.

        Returns
        -------
        None

        """
        self.image_shpae = image.shape
        pad = (self.size-1)//2
        image = F.pad(input=image, pad=[pad] * 4, mode='constant', value=0)
        
        result = torch.Tensor(self.image_shpae[0]//self.stride, self.image_shpae[1]// self.stride)
        x = 0
        for i in range(result.shape[0]):
            y = 0
            for j in range(result.shape[1]):
                result[i][j] = torch.sum(image[x:x+self.size, y:y+self.size] * self.filter)
                y += self.stride
            x += self.stride
        self.result = result
        return result
    
class DoGFilter(AbstractFilter):
    def __init__(
            self,
            sigma1: float = 3,
            sigma2: float = 7,
            size: int = 9,
            center_type: str = "on",
            stride: int = 1,
            **kwargs
    ) -> None:
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            super().__init__(size=size, center_type=center_type)
            
    def DoG_function(self, x, y, **kwargs) -> float:
        x = self.size // 2 - x
        y = self.size // 2 - y
        result = 1/np.sqrt(2*np.pi) * ((1/self.sigma1) * (np.e ** \
                                       (-(x**2 + y**2)/(2*self.sigma1**2))) -\
                                    (1/self.sigma2) * (np.e ** \
                                        (-(x**2 + y**2)/(2*self.sigma2**2))))
        return self.control * result
    
    def filter_maker(self, **kwargs) -> torch.Tensor:
        f = torch.Tensor(self.size, self.size)
        for i in range(self.size):
            for j in range(self.size):
                f[i][j] = self.DoG_function(j, i)
         
        return f - (torch.sum(f) / (self.size ** 2))

    def save_filterPic(self, name: str) -> None:
        
        super().save_filterPic(name=name if name != None else "DoGFilter")
        
    def save_outputPic(self, name: str) -> None:
        
        super().save_outputPic(name=name if name != None else "DoG_Result")
        
    def __str__(self) -> torch.Tensor:
        
        return super().__str__()
    
    def __call__(self, image: torch.Tensor, **kwargs) -> torch.Tensor:

        return super().__call__(image)
    
class GaborFilter(AbstractFilter):
    def __init__(
            self,
            landa: float = 20.,
            theta: float = np.pi/2,
            sigma: float = 15.,
            gamma: float = .1,
            size: int = 9,
            center_type: str = "on",
            stride: int = 1,
            **kwargs
    ) -> None:
            self.landa = landa
            self.theta = theta
            self.sigma = sigma
            self.gamma = gamma
            
            super().__init__(size=size, center_type=center_type)
        
    def Gabor_function(self, x, y, **kwargs) -> float:
        x = self.size // 2 - x
        y = self.size // 2 - y
        X = x * np.cos(self.theta) + y * np.sin(self.theta)
        Y = -x * np.sin(self.theta) + y * np.cos(self.theta)
        result = np.e ** (-(X**2+ self.gamma**2 * Y**2)/(2* self.sigma**2)) * np.cos(2*np.pi * X/self.landa)
        
        return result * self.control
    
    def filter_maker(self, **kwargs) -> torch.Tensor:
        f = torch.Tensor(self.size, self.size)
        for i in range(self.size):
            for j in range(self.size):
                f[i][j] = self.Gabor_function(i, j)
       
        return f - (torch.sum(f) / (self.size ** 2))
    
    def save_filterPic(self, name: str) -> None:
        
        super().save_filterPic(name=name if name != None else "GaborFilter")

    def save_outputPic(self, name: str) -> None:
        
        super().save_outputPic(name=name if name != None else "Gabor_Result")

    def __str__(self) -> torch.Tensor:
        
        return super().__str__()
    
    def __call__(self, image, **kwargs) -> torch.Tensor:
        return super().__call__(image)