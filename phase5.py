from cnsproject.encoding.encoders import *
from cnsproject.plotting.plotting import raster
import cv2
import torch

img = cv2.imread('img/001.jpg')

img = torch.from_numpy(img)
img = img.sum(2)//3

m = PoissonEncoder(100, r=0.15)
encode = m(img)
encode = encode.view(encode.size(0), encode.size(1)*encode.size(2))
raster(encode)

m = Time2FirstSpikeEncoder(100)
encode = m(img)
encode = encode.view(encode.size(0), encode.size(1)*encode.size(2))
raster(encode)


m = PositionEncoder(100, node_n=100, std=0.4)
encode = m(img)
raster(encode)
