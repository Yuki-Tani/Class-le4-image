import ioex
import layer
import util
import nn
import sys
import matplotlib.pyplot as plt
from pylab import cm

import advanced_base as base

SEED = 100

print("### advanced8 [Convolution Pooling] ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
neuralNet = nn.NeuralNetwork1C1P3LReLU(28*28,4,3,2,50,10,SEED)

base.runMainController(neuralNet,inputM,outputM)

print("---conv, pool check---")
outputM.showPictureFromBatch(neuralNet.convLayer.getOutput(),(28*4,28))
outputM.showPictureFromBatch(neuralNet.poolLayer.getOutput(),(28*2,14))
