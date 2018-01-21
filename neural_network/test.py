import ioex
import util
import layer
import data
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import time

SAMPLE_IMAGE = np.array([[0,100,0],[100,200,100],[0,100,0]])
SAMPLE_WEIGHT = np.identity(9) * 0.01
SAMPLE_SHIFT = np.array([[1],[0],[1],[0],[-1],[0],[1],[0],[1]])

SAMPLE_OUTPUT = np.array(
    [[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]])

class DriverLayer(layer.Layer):
    def __init__(self):
        super().__init__(SAMPLE_OUTPUT.size)

    def calculate(self):
        return SAMPLE_OUTPUT

def inputTest():
    inputM = ioex.InputManager()
    trainingData = inputM.getMnistTrainingData()

    print(trainingData.images.shape)
    print(trainingData.answers.shape)
    # trainingData.shuffle()
    sample = trainingData.getImageBatch(3)
    print(sample)

    print(trainingData.images.shape)
    print(trainingData.answers.shape)

    idx = inputM.selectNumber()
    sample = trainingData.getSingleData(idx)

    print("answer: " + str(sample.answer))
    print(sample.getSize())

    print("show picture?")
    selected = inputM.selectYesOrNo()
    if selected :
        plt.imshow(sample.image, cmap=cm.gray)
        plt.show()
    return sample

def inputLayerTest():
    sample = data.MnistData(SAMPLE_IMAGE,0)
    inputLayer = layer.InputLayer(sample.getSize())
    image = sample.getImage()
    inputLayer.setInput(image)
    inputLayer.confirmParameters()
    return inputLayer

def hiddenLayerTest():
    prev = layer.Layer(10)
    hiddenLayer = layer.HiddenLayer(5,prev)
    hiddenLayer.confirmParameters()

def inputToHiddenTest():
    inputLayer = inputLayerTest()
    hiddenLayer = layer.HiddenLayer(9,inputLayer)
    hiddenLayer.setWeight(SAMPLE_WEIGHT)
    hiddenLayer.setShift(SAMPLE_SHIFT)
    print(hiddenLayer.calculate())

def outputLayerTest():
    driver = DriverLayer()
    outputLayer = layer.OutputLayer(9,driver)
    print(outputLayer.calculate())
    outputLayer.setActivator(util.correctedSoftmax)
    print(outputLayer.calculate())
    outputLayer.setWeight(outputLayer.maker.getIdentityComplex(
        outputLayer.weightSize
    ))
    outputLayer.setShift(np.zeros(outputLayer.shiftSize))
    print(outputLayer.calculate())
    #print(outputLayer.getOutput())
    return outputLayer.getOutput()

def outputTest():
    outputM = ioex.OutputManager()
    outputM.printMaxLikelihood(SAMPLE_OUTPUT)

def calcLossTest():
    driver = DriverLayer()
    outputLayer = layer.OutputLayer(9,driver)
    out = outputLayer.calculate()
    label = np.array([[0],[0],[0],[1],[0],[0],[0],[0],[0]])
    print(out)
    print(label)
    print(util.crossEntropy(out,label))

def dataTest():
    inputM = ioex.InputManager()
    trainingData = inputM.getMnistTrainingData()

    print("\ngetImageBatch")
    for i in range(1,11):
        #計測スタート
        start = time.time()
        batch = trainingData.getImageBatch(50*i)
        #print(batch)
        #print(batch.shape)
        #計測終了
        elapsed_time = time.time() - start
        print(50*i,end=": ")
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print("\ngetAnswerVectorBatch")
    for i in range(1,11):
        #計測スタート
        start = time.time()
        batch = trainingData.getAnswerVecotrBatch(50*i)
        #計測終了
        elapsed_time = time.time() - start
        print(50*i,end=": ")
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print("\nshaffle")
    #計測スタート
    start = time.time()
    trainingData.shuffle()
    #計測終了
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

def inputConvTest():
    inputM = ioex.InputManager()
    outputM = ioex.OutputManager()
    trainingData = inputM.getMnistTrainingData()
    sample = trainingData.getImageBatch(5)
    outputM.showPictureFromBatch(sample,(28,28))

    inputLayer = layer.InputLayer(28*28)
    conv = layer.ConvolutionalLayer(4,3,inputLayer)
    outputLayer = layer.OutputLayer(10,conv)

    #conv.setWeight(np.array([[1,1,1,1,-8,1,1,1,1],[1,0,1,0,-4,0,1,0,1],[0,1,0,1,-4,1,0,1,0],[0,0,0,0,1,0,0,0,0]]))
    inputLayer.setInputBatch(sample)

    for i in range(0,10000):
        outputLayer.calculate()
        if i%1000 == 0 :
            outputM.showPictureFromBatch(conv.getOutput(),(28*4,28))
            print(conv.getWeight())
        outputLayer.update(trainingData.getAnswerVecotrBatch(5))

#inputConvTest()

"""
testInput = np.array([[11,2,3,4,5],[12,2,3,4,5],[13,2,3,4,5],[14,2,3,4,5],[21,2,3,4,5],[22,2,3,4,5],[23,2,3,4,5],[24,2,3,4,5],[31,2,3,4,5],[32,2,3,4,5],[33,2,3,4,5],[34,2,3,4,5],[41,2,3,4,5],[42,2,3,4,5],[43,2,3,4,5],[44,2,3,4,5]])

inputLayer = layer.InputLayer(16)
conv = layer.ConvolutionalLayer(4,3,inputLayer)
inputLayer.normalization = 1
inputLayer.setInputBatch(testInput)
result = conv.calculate()

print(result)
"""


dEdOutTest = np.array( [[11,2,3,4,5],[12,2,3,4,5],[13,2,3,4,5],[14,2,3,4,5],[21,2,3,4,5],[22,2,3,4,5],[23,2,3,4,5],[24,2,3,4,5],[31,2,3,4,5],[32,2,3,4,5],[33,2,3,4,5],[34,2,3,4,5],[41,2,3,4,5],[42,2,3,4,5],[43,2,3,4,5],[44,2,3,4,5],[11,2,3,4,5],[12,2,3,4,5],[13,2,3,4,5],[14,2,3,4,5],[21,2,3,4,5],[22,2,3,4,5],[23,2,3,4,5],[24,2,3,4,5],[31,2,3,4,5],[32,2,3,4,5],[33,2,3,4,5],[34,2,3,4,5],[41,2,3,4,5],[42,2,3,4,5],[43,2,3,4,5],[44,2,3,4,5],[11,2,3,4,5],[12,2,3,4,5],[13,2,3,4,5],[14,2,3,4,5],[21,2,3,4,5],[22,2,3,4,5],[23,2,3,4,5],[24,2,3,4,5],[31,2,3,4,5],[32,2,3,4,5],[33,2,3,4,5],[34,2,3,4,5],[41,2,3,4,5],[42,2,3,4,5],[43,2,3,4,5],[44,2,3,4,5],[11,2,3,4,5],[12,2,3,4,5],[13,2,3,4,5],[14,2,3,4,5],[21,2,3,4,5],[22,2,3,4,5],[23,2,3,4,5],[24,2,3,4,5],[31,2,3,4,5],[32,2,3,4,5],[33,2,3,4,5],[34,2,3,4,5],[41,2,3,4,5],[42,2,3,4,5],[43,2,3,4,5],[44,2,3,4,5]])

#conv.update(dEdOutTest)

inputLayer = layer.InputLayer(64)
inputLayer.normalization = 1
pooling = layer.PoolingLayer(2,(16,4),inputLayer)
inputLayer.setInputBatch(dEdOutTest)

result = pooling.calculate()
print(result)

pooling.update(result)

"""
t1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
t2 = np.array([1,2,9])

print(t1.reshape(2,2,3)*t2)
print(t1.reshape(2,2,3)==t2)

dt = np.zeros(t1.shape)
print(dt)
dt[t1==t2] = 1
print(dt)
"""

"""
testOutput = np.array([[11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]])

result = conv.postCalculate(testOutput)
print(result)
"""
"""
dim2a = np.array(
[[10,10,10,1,1,1,10,10,10,1,1,1,10,10,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
 [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
 [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
 [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
 [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]]
)
dim2b = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])

print(dim2a[1:-1,1:-1])

stage1 = dim2a.reshape((-1,2,3,2,3))
print(stage1)
stage2 = stage1.transpose((0,1,3,2,4))
print(stage2)
stage3 = stage2.reshape((-1,4,9))
print(stage3)

print(stage3.shape[2])
"""
"""
ar = np.array([[1,-2],[-7,1],[0,0]])
print(util.relu(ar))
print(util.sigmoid(ar))
"""
#dataTest()

"""
print(np.zeros((10,1)))
print(np.zeros((1,10)))
print(np.zeros((10,2)))
print(np.zeros((2,10)))
"""
"""
output = np.array([[100,2,3],[4,500,6],[7,800,9],[10,11,12]])
batch = np.array([[1,0,0],[0,0,0],[0,1,1],[0,0,0]])

result = output.argmax(axis = 0)
print(result)
answer = batch.argmax(axis = 0)
print(answer)
hit = (result == answer)
print(hit)
result[True] = 0
print(result)
result[hit] = 1
print(result.mean() * 100)
"""

"""
print(SAMPLE_OUTPUT)
read = np.random.shuffle(SAMPLE_OUTPUT)
print(SAMPLE_OUTPUT)
print(read)
"""
#inputTest()
#inputLayerTest()
#shiddenLayerTest()
#inputToHiddenTest()
#outputLayerTest()
#outputTest()
#calcLossTest()
"""
three = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]]])
three = three[0:2]
two = three.reshape((2,-1)).T
print(two)
"""
#a = np.array([[2,3],[4,5],[6,7]])
#print(a - 1)
#print(a.size)

#b = np.array([[10],[3],[1]])
#print(util.softmax(b))
#print(util.correctedSoftmax(b))

#c = a + b
#print(c)
#print(c.sum())
#print(c.sum(axis=0))
#print(c.sum(axis=1))

#d = np.array([[1,2],[3,4],[5,6]])
#e = np.array([[7],[8],[9]])
#f = np.append(d,e)
#print(f)
#f = np.append(d,e,axis = 0)
#print(f)
#f = np.append(d,e,axis = 1)
#print(f)
