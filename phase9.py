import cv2
import torch
import numpy as np
from cnsproject.encoding.encoders import *
from cnsproject.plotting.plotting import raster

from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.connections import ConvolutionalConnection, PoolingConnection
from cnsproject.network.monitors import Monitor
from cnsproject.learning.learning_rules import STDP

from cnsproject.plotting.plotting import raster

from cnsproject.encoding.filters import *

img = cv2.imread('img/001.jpg')

img = torch.from_numpy(img)
img = img.sum(2)//3


time = 10
sigma1 = [1,2]
sigma2 = [4,5]
filter_size = [7, 7] # DoG filter size is the same
pooling_size = [4, 4]
features = 2

stride_conv = 1
stride_pool = 4
nima = 0 
vahdat = 0
def pic(x, name=""):
    global img, nima, vahdat, features
    feature = 0
    for i in x:
        i *= 255
        i = abs(i - 255)
        pic = np.array(i)
        pic = cv2.resize(pic, (img.shape[1], img.shape[0]))
        cv2.imwrite("feature"+str(feature)+"_time"+str(vahdat)+"_"+name+".jpg", pic)
        feature += 1
    vahdat += 1
    if vahdat % 2*features == 0:
        nima += 1


def DoG(sigma1, sigma2):
    global filter_size
    dog = DoGFilter(size=filter_size[0], sigma1=sigma1, sigma2=sigma2, center_type="on")
    return dog.filter

def Time2First():
    global img, time
    encode = Time2FirstSpikeEncoder(time)
    return encode(img)

def main():
    global time, features, filter_size, sigma1, sigma2, stride_conv
    
    code = Time2First()
    
    w = torch.Tensor(features, *filter_size)
    for i in range(features):
        w[i] = DoG(sigma1[i], sigma2[i])
    
    input_layer = InputPopulation(shape=code[0].shape, dt=1)
    middle_layer = LIFPopulation(shape=(features, code.shape[1]//stride_conv, code.shape[2]//stride_conv), threshold=-69.8)
    out_layer = LIFPopulation(shape=(features, code.shape[1]//(stride_conv*stride_pool), code.shape[2]//(stride_conv*stride_pool)), threshold=-69.1)
    
    connect_in_mid = ConvolutionalConnection(pre=input_layer, post=middle_layer, 
                                             filter_size=filter_size, feature_n=features,
                                             stride=stride_conv, w=w)

    connect_mid_out = PoolingConnection(pre=middle_layer, post=out_layer, pooling_size=pooling_size, stride=stride_pool)    
    
    monitor_mid = Monitor(middle_layer, state_variables=["s"])
    monitor_mid.set_time_steps(time, dt=1)
    monitor_mid.reset_state_variables()
    
    monitor_out = Monitor(out_layer, state_variables=["s"])
    monitor_out.set_time_steps(time, dt=1)
    monitor_out.reset_state_variables()

    for i in range(time):
        # pic(effect_inp[0], name="conv")
        # pic(effect_mid[0], name="pool")
        
        current = 0
        input_layer.forward(code[i])

        effect_inp = connect_in_mid.compute(input_layer.s)

        middle_layer.forward(current, effect_inp)
        effect_mid = connect_mid_out.compute(middle_layer.s)
        out_layer.forward(0, effect_mid)
        
        # pic(middle_layer.s[0].float(), name="midSpike")
        # pic(out_layer.s[0].float(), name="outSpike")
        
        monitor_mid.record()
        monitor_out.record()
        
        pic(middle_layer.s.float(), name="middle")
        # pic(out_layer.s.float(), name="out")
    
    output = torch.flatten(monitor_out.get('s'), 1, 3)
    middle = torch.flatten(monitor_mid.get('s'), 1, 3)
    
    raster(middle)    
    raster(output)    

main()