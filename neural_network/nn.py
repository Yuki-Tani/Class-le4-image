import numpy as np
import layer
import util
import learning

class NeuralNetwork3Layers :
    def __init__(self, inputDim, hiddenDim, outputDim, seed = 1):

        # 三層ニューラルネットワーク
        self.inputLayer = layer.InputLayer(inputDim)
        self.hiddenLayer = layer.HiddenLayer(hiddenDim,self.inputLayer,seed)
        self.outputLayer = layer.OutputLayer(outputDim,self.hiddenLayer,seed)
        # 正解データ
        self.answer = np.zeros(outputDim)

    ## 入力系メソッド ###

    def setInput(self, inputData):
        self.inputLayer.setInput(inputData)

    def setInputBatch(self, inputBatch):
        self.inputLayer.setInputBatch(inputBatch)

    def setAnswer(self, answerVector):
        self.answer = answerVector.reshape(answerVector.size,1)
        if(self.answer.size != self.outputLayer.dimension):
            print("WARNING! Input data is NOT match to deimension")

    def setAnswerBatch(self,answerBatch):
        self.answer = answerBatch

    ## 活性メソッド ##

    def calculate(self):
        return self.outputLayer.calculate()

    def getLoss(self):
        return self.outputLayer.getLoss(self.answer)

    def getPersentageOfCurrent(self):
        result = self.outputLayer.getOutput().argmax(axis = 0)
        ans = self.answer.argmax(axis = 0)
        hit = (result == ans)
        result[True] = 0
        result[hit] = 1
        return result.mean() * 100

    def update(self, updateRatio = 0.01):
        self.outputLayer.update(self.answer, updateRatio)

    def learn(self, imageBatch, answerBatch, updateRatio = 0.01):
        self.setInputBatch(imageBatch)
        self.setAnswerBatch(answerBatch)
        self.calculate()
        self.update(updateRatio)

    def save(self,
        fileHW = "./learned_data/hiddenWeight",
        fileHS = "./learned_data/hiddenShift",
        fileOW = "./learned_data/outputWeight",
        fileOS = "./learned_data/outputShift") :
        """
        self.hiddenLayer.confirmParameters()
        self.outputLayer.confirmParameters()
        """
        np.save(fileHW, self.hiddenLayer.weight)
        np.save(fileHS, self.hiddenLayer.shift)
        np.save(fileOW, self.outputLayer.weight)
        np.save(fileOS, self.outputLayer.shift)
        print("saved.")

    def load(self,
        fileHW = "./learned_data/hiddenWeight.npy",
        fileHS = "./learned_data/hiddenShift.npy",
        fileOW = "./learned_data/outputWeight.npy",
        fileOS = "./learned_data/outputShift.npy") :
        """
        print("\n >> BEFORE")
        self.hiddenLayer.confirmParameters()
        self.outputLayer.confirmParameters()
        """
        hw = np.load(fileHW)
        hs = np.load(fileHS)
        ow = np.load(fileOW)
        os = np.load(fileOS)
        self.hiddenLayer.setWeight(hw)
        self.hiddenLayer.setShift(hs)
        self.outputLayer.setWeight(ow)
        self.outputLayer.setShift(os)
        """
        print("\n >> AFTER")
        self.hiddenLayer.confirmParameters()
        self.outputLayer.confirmParameters()
        """
        print("loaded.")

class NeuralNetwork3LayersReLU(NeuralNetwork3Layers) :
    def __init__(self, inputDim, hiddenDim, outputDim, seed = 1):

        # 三層ニューラルネットワーク
        self.inputLayer = layer.InputLayer(inputDim)
        self.hiddenLayer = layer.HiddenLayer(hiddenDim,self.inputLayer,seed)
        self.outputLayer = layer.OutputLayer(outputDim,self.hiddenLayer,seed)

        self.hiddenLayer.setActivator(util.relu)
        self.hiddenLayer.setBackPropagator(learning.backPropReLU)

        # 正解データ
        self.answer = np.zeros(outputDim)
