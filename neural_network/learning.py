import numpy as np
import util

#逆伝播関数((ConversionLayer, dEdOut) -> (dEdIn, dEdWeight, dEdShift))
def backPropFullyConnect(conversionLayer,dEdOut):
    weight = conversionLayer.weight
    layerInput = conversionLayer.getInput()
    dEdIn = (weight.T).dot(dEdOut)
    dEdWeight = dEdOut.dot(layerInput.T)
    dEdShift = dEdOut.sum(axis = 1, keepdims = True)

    #dEdWeight = dEdOut.dot(layerInput.T) / dEdOut.shape[1]
    #dEdShift = dEdOut.mean(axis = 1, keepdims = True)

    #print("O, I, W, S [learning 11]")
    #print(dEdOut)
    #print(dEdIn)
    #print(dEdShift)
    return dEdIn, dEdWeight, dEdShift

def backPropIdentity(conversionLayer,dEdOut):
    dEdActivatorIn = dEdOut
    return backPropFullyConnect(conversionLayer, dEdActivatorIn)

def backPropSigmoid(conversionLayer, dEdOut):
    activatorOut = conversionLayer.getOutput()
    dEdActivatorIn = (1-activatorOut)*activatorOut*dEdOut
    return backPropFullyConnect(conversionLayer, dEdActivatorIn)

def backPropReLU(conversionLayer, dEdOut):
    activatorOut = conversionLayer.getOutput()
    activatorOut[activatorOut>0] = 1 #dReLU
    dEdActivatorIn = activatorOut*dEdOut
    return backPropFullyConnect(conversionLayer, dEdActivatorIn)

def backPropSoftmaxAndCrossEntropy(conversionLayer, answerMatrix):
    activatorOut = conversionLayer.getOutput()
    batchSize = answerMatrix.shape[1]
    dEdActivatorIn = (activatorOut - answerMatrix) / batchSize
    return backPropFullyConnect(conversionLayer, dEdActivatorIn)

#更新関数
#確率的勾配降下法
def updateBySGD(conversionLayer, dEdWidth , dEdShift, learningRatio = 0.01) :
    newWeight = conversionLayer.weight - learningRatio * dEdWidth
    newShift = conversionLayer.shift - learningRatio * dEdShift
    conversionLayer.setWeight(newWeight)
    conversionLayer.setShift(newShift)
