import numpy as np
import util
import learning

##################################################
class Layer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.output = np.zeros((dimension, 1))
        self.lossFunction = util.crossEntropy

    def calculate(self):
        pass

    def update(self, dEdOut):
        pass

    def getOutput(self):
        return self.output

    def confirmParameters(self):
        print("[Layer]")
        print("dimension : "+str(self.dimension))

    def getDimension(self):
        return self.dimension

    def getLoss(self, answer):
        return self.lossFunction(self.getOutput(),answer)

##################################################
class InputLayer(Layer):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.input = np.zeros(dimension)
        self.inputBatch = np.empty((dimension,0))
        self.normalization = 256

    def setInput(self, inputData):
        self.input = inputData.reshape(inputData.size,1)
        if(self.input.size != self.dimension):
            print("WARNING! Input data is NOT match to deimension")

    def setInputBatch(self, inputBatch):
        self.input = inputBatch
        if(self.input.shape[0] != self.dimension):
            print("WARNING! Input batch is NOT match to deimension")
    #バッチ入力の開始
    def initInputBatch(self):
        self.inputBatch = np.empty((self.dimension,0))

    #バッチデータの登録
    def addInputBatchData(self, inputData):
        addData = inputData.reshape(inputData.size,1)
        if(addData.size != self.dimension):
            print("WARNING! Input data is NOT match to deimension")
        else:
            self.inputBatch = np.hstack((self.inputBatch,addData))

    #バッチデータ登録完了
    def finishInputBatch(self):
        self.input = self.inputBatch

    def calculate(self):
        self.output = self.input / self.normalization
        return self.output

    def confirmParameters(self):
        print("[Input Layer]")
        print("dimension : "+str(self.dimension))
        print("input data size : "+str(self.input.size))

##################################################
class ConversionLayer(Layer):
    def __init__(self, dimension, prevLayer):
        super().__init__(dimension)
        self.prevLayer = prevLayer
        self.maker = util.ComplexMaker()
        self.weightSize = (dimension, prevLayer.getDimension())
        self.shiftSize = (dimension,1)
        self.weight = self.maker.getIdentityComplex(self.weightSize)
        self.shift = np.zeros(self.shiftSize)
        #結合関数
        self.connector = util.fullyConnect
        #活性化関数
        self.activator = util.identity
        #逆伝播関数(layer, dE/dOut -> dE/dIn, dE/dWeight, dE/dShift)
        self.backPropagator = learning.backPropIdentity
        #更新関数(layer, dE/dWidth, dE/dShift, learningRatio)
        self.updator = learning.updateBySDG

    def getInput(self):
        return self.prevLayer.getOutput()

    def calculate(self):
        inputVector = self.prevLayer.calculate()
        value = self.connector(inputVector, self.weight, self.shift)
        self.output = self.activator(value)
        return self.output

    def update(self, dEdOut, learningRatio = 0.01):
         dEdIn, dEdWidth, dEdShift = self.backPropagator(self, dEdOut)
         self.updator(self, dEdWidth, dEdShift, learningRatio)
         self.prevLayer.update(dEdIn)

    def setWeight(self,weight):
        if(weight.shape == self.weightSize):
            self.weight = weight
        else:
            print("WARNING! This weight is NOT match to matrix shape.")

    def setShift(self,shift):
        #print("shift [layer,112]")
        #print(shift)
        if(shift.shape == self.shiftSize):
            self.shift = shift
        else:
            print("WARNING! This shift is NOT match to matrix shape.")

    def setActivator(self,function):
        if(type(self.activator) == type(function)):
            self.activator = function
        else:
            print("WARNING! This activator is NOT function")

    def setConnector(self,function):
        if(type(self.connector) == type(function)):
            self.connector = function
        else:
            print("WARNING! This connector is NOT function")

    def setBackPropagator(self, function) :
        if(type(self.backPropagator) == type(function)):
            self.backPropagator = function
        else:
            print("WARNING! This backPropagator is NOT function")

    def setUpdator(self, function) :
        if(type(self.updator) == type(function)):
            self.updator = function
        else:
            print("WARNING! This updator is NOT function")

    def confirmParameters(self):
        print("[Conversion Layer]")
        print("dimension : "+str(self.dimension))
        print("weight :")
        print(self.weight)
        print("shift :")
        print(self.shift)

##################################################
class HiddenLayer(ConversionLayer):
    def __init__(self, dimension, prevLayer, weightSeed = 1, shiftSeed = 1):
        super().__init__(dimension,prevLayer)
        randomScale = 1.0 / prevLayer.getDimension()
        self.maker.setSeed(weightSeed)
        self.setWeight(
            self.maker.getRandomComplex(self.weightSize, randomScale)
        )
        self.maker.setSeed(shiftSeed)
        self.setShift(
            self.maker.getRandomComplex(self.shiftSize, randomScale)
        )
        self.setActivator(util.sigmoid)
        self.setBackPropagator(learning.backPropSigmoid)

    def confirmParameters(self):
        print("[Hidden Layer]")
        print("dimension : "+str(self.dimension))
        print("weight :")
        print(self.weight)
        print("shift :")
        print(self.shift)

##################################################
class OutputLayer(ConversionLayer):
    def __init__(self, dimension, prevLayer, weightSeed = 1, shiftSeed = 1):
        super().__init__(dimension, prevLayer)
        randomScale = 1.0 / prevLayer.getDimension()
        self.maker.setSeed(weightSeed)
        self.setWeight(
            self.maker.getRandomComplex(self.weightSize, randomScale)
        )
        self.maker.setSeed(shiftSeed)
        self.setShift(
            self.maker.getRandomComplex(self.shiftSize, randomScale)
        )
        self.setActivator(util.softmax)
        self.setBackPropagator(learning.backPropSoftmaxAndCrossEntropy)

    def update(self, answerMatrix, learningRatio = 0.01):
        dEdIn, dEdWidth, dEdShift = self.backPropagator(self, answerMatrix)
        self.updator(self, dEdWidth, dEdShift, learningRatio)
        self.prevLayer.update(dEdIn)

    def confirmParameters(self):
        print("[Output Layer]")
        print("dimension : "+str(self.dimension))
        print("weight : ")
        print(self.weight)
        print("shift : ")
        print(self.shift)
