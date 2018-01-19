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

ar = np.array([[1,-2],[-7,1],[0,0]])
print(util.relu(ar))
print(util.sigmoid(ar))

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
