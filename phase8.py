import cv2
import torch
import numpy as np
from cnsproject.encoding.encoders import *
from cnsproject.plotting.plotting import raster

from cnsproject.encoding.filters import *

img = cv2.imread('img/chandler.jpg')

img = torch.from_numpy(img)
img = img.sum(2)//3

def DoG(size, sigma1, sigma2, center_type):
    global img
    dog = DoGFilter(size=size, sigma1=sigma1, sigma2=sigma2, center_type=center_type)
    dog.save_filterPic("results/DoG_Filter")
    
    # Uncomment to see the filter tensor
    # print(dog)
    
    result = dog(img)
    
    # Uncomment to see the result tensor
    # print(result)
    
    dog.save_outputPic(name="results/DoG_out")
    
    return result
    
def Gabor(size, landa, theta, sigma, gamma, center_type):
    global img
    gabor = GaborFilter(size=size, landa=landa, theta=theta, sigma=sigma, gamma=gamma, center_type=center_type)
    gabor.save_filterPic("results/Gabor_Filter")
    
    print(gabor.filter)
    # Uncomment to see the filter tensor
    # print(gabor)    
    
    result = gabor(img)
    
    # Uncomment to see the result tensor
    # print(result)

    gabor.save_outputPic(name="results/Gabor_out")
    
    return result

def Time2First(result, name):
    global img
    encode = Time2FirstSpikeEncoder(100)
    spike = encode(result)
    
    spike *= 255
    spike = abs(spike - 255)
    pic = np.array(spike[-2])
    pic = cv2.resize(pic, (img.shape[1], img.shape[0]))
    cv2.imwrite(name + "_time2first.jpg", pic)

def Poisson(result, name):
    global img
    encode = PoissonEncoder(100, r=0.7)
    spike = encode(result)
    
    spike *= 255
    # spike = abs(spike - 255)
    pic = np.array(spike[0])
    pic = cv2.resize(pic, (img.shape[1], img.shape[0]))
    cv2.imwrite(name + "_poisson.jpg", pic)    

dog_result = DoG(9, 2, 9, 'on')
gabor_result = Gabor(5, 10, 0, 5, 0.1, 'on') 

Time2First(dog_result, "results/DoG")
Poisson(dog_result, "results/DoG")

Time2First(gabor_result, "results/Gabor")
Poisson(gabor_result, "results/Gabor")