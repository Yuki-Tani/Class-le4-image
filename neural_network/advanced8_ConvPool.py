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
"""
neuralNet = nn.NeuralNetwork1C1P3LReLU(
                28*28,
                4,3,(28,28),
                2,(28*4,28),
                50,
                10,
                SEED)
"""
print("input filterNum")
filNum = inputM.selectNumber()
print("input hidden dimension")
hidden = inputM.selectNumber()

neuralNet = nn.NeuralNetwork1C1P3LReLU(
                28*28,
                filNum,3,(28,28),
                2,(28*filNum,28),
                hidden,
                10,
                SEED)

"""
neuralNet = nn.NeuralNetwork2C2P3LReLU(
                28*28, #input
                4,3,(28,28), #conv1
                2, # pool1
                4,3,(2*28,14), #conv2
                2, # pool2
                50, # hidden
                10, # output
                SEED)
"""

base.runMainController(neuralNet,inputM,outputM)

"""
print("---conv, pool check---")
#outputM.showPictureFromBatch(neuralNet.convLayer.getOutput(),(28*4,28))
#outputM.showPictureFromBatch(neuralNet.poolLayer.getOutput(),(28*2,14))

outputM.showPictureFromBatch(neuralNet.convLayer.getOutput(),(28*8,28))
outputM.showPictureFromBatch(neuralNet.poolLayer.getOutput(),(14*8,14))

#outputM.showPictureFromBatch(neuralNet.convLayer1.getOutput(),(28*4,28))
#outputM.showPictureFromBatch(neuralNet.poolLayer1.getOutput(),(28*2,14))
#outputM.showPictureFromBatch(neuralNet.convLayer2.getOutput(),(28*8,14))
#outputM.showPictureFromBatch(neuralNet.poolLayer2.getOutput(),(28*4,7))
"""
