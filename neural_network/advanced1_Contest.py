import ioex
import layer
import util
import nn
import sys
import matplotlib.pyplot as plt
from pylab import cm

import advanced_base as base

SEED = 100

print("### advanced1 [Contest] ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()

print("input hidden dimension")
hidden = inputM.selectNumber()

neuralNet = nn.NeuralNetwork3LayersReLU(28*28, hidden, 10, SEED)

base.runMainController(neuralNet,inputM,outputM)
