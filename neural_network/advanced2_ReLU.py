import ioex
import layer
import util
import nn
import sys
import matplotlib.pyplot as plt
from pylab import cm

import advanced_base as base

HIDDEN_DIMENSION = 50
SEED = 100

print("### advanced2 [ReLU] ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
neuralNet = nn.NeuralNetwork3LayersReLU(28*28, HIDDEN_DIMENSION, 10, SEED)

base.runMainController(neuralNet,inputM,outputM)
