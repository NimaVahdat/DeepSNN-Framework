import torch
import cv2
import torch
import numpy as np
from cnsproject.encoding.encoders import *
from cnsproject.plotting.plotting import raster

from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.connections import ConvolutionalConnection
from cnsproject.network.monitors import Monitor
from cnsproject.learning.learning_rules import STDP

from cnsproject.utils import Lateral_inhibition1, Lateral_inhibition2
from cnsproject.decision.decision import WinnerTakeAllDecision

from cnsproject.plotting.plotting import raster

from cnsproject.encoding.filters import *

time = 10

# =============================================================================
# DoG Filter 
# =============================================================================
size = 3
sigma1 = 1
sigma2 = 10
center_type = 'on'
dog = DoGFilter(size=size, sigma1=sigma1, sigma2=sigma2, center_type=center_type)

# =============================================================================
# Time to First spike
# =============================================================================
encode = Time2FirstSpikeEncoder(time)


# =============================================================================
# parameters
# =============================================================================
features_number = 4
features_kernel_size = [5, 5]
stride = 1
lr = [0.05, 0.11]
wmin = 0
wmax = 1
k = 4

sigma = 2

neighbor = 3
coefficient = 5

w = torch.clamp(torch.rand(features_number, 1, *features_kernel_size), wmin, wmax)

ww = torch.Tensor(40, features_number, 1, *features_kernel_size)

training_pic_num = 40
for i in range(1, training_pic_num):
    print("*********************", i)
    # img = cv2.imread('img/face.jpg')
    img = cv2.imread('caltech2/image_%.4d.jpg'%(i))
    # img = cv2.imread('caltech_motor/%.4d.jpg'%(i))
    img = cv2.resize(img, (130, 160))
    # print('caltech/image_%.4d.jpg'%(i))
    img = torch.from_numpy(img)
    img = img.sum(2)//3
    
    result_DoG = dog(img)
    # dog.save_outputPic('DOG_%.4d.jpg'%i)
    input_spike = encode(result_DoG)
    
    time, height, width = input_spike.shape
    
    # Layers
    input_layer = InputPopulation(shape=(height, width), dt=1, tau_s=5)
    output_layer = LIFPopulation(shape=(1, features_number, height//stride, width//stride), threshold=-65, tau=10, tau_s=10)
    
    # Connection
    connection = ConvolutionalConnection(pre=input_layer, post=output_layer, 
                                                 filter_size=features_kernel_size, feature_n=features_number,
                                                 stride=stride, lr=lr, learning_rule=STDP, dt=1.0, w=w)
    
    monitor = Monitor(output_layer, state_variables=["s"])
    monitor.set_time_steps(time, dt=1)
    monitor.reset_state_variables()
    
    kwinner = WinnerTakeAllDecision(k=k, shape=(1, features_number, height//stride, width//stride))
    
    for j in range(time):
        input_layer.forward(input_spike[j])
        effect = connection.compute(input_layer.s)
        output_layer.v += Lateral_inhibition1(output_layer.s.float(), neighbor, sigma, coefficient) 
        # print(Lateral_inhibition1(output_layer.s.float(), neighbor, sigma, coefficient).sum())
        output_layer.forward(traces=effect)
        
        
        
        control = kwinner.compute(output_layer.s.float(), output_layer.v)
        # output_layer.v += Lateral_inhibition2(output_layer.s.float(), neighbor, sigma, coefficient) 
        output_layer.refractory_and_reset()
        # print(torch.min(output_layer.v), torch.max(output_layer.v))
        connection.update(control=control)

        monitor.record()
    
    w = connection.w
    ww[i] = w
    s = monitor.get('s')

    # print(w)
    # features_pic(w)

def features_pic(w):
    w = torch.squeeze(w)
    w -= w.min()
    w = (w/w.max()) * 255
    for i in range(w.shape[0]):
        pic = cv2.resize(np.array(w[i]), (100, 100))
        cv2.imwrite("feature_%.2d.jpg"%i, pic)
features_pic(ww[30])