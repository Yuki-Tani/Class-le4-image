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

    def getInfo(self) :
        return "NeuralNetwork3Layers\n"+"hidden layer:  "+str(self.hiddenLayer.getDimension())

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

    def getInfo(self) :
        return "NeuralNetwork3LayersReLU\n"+"hidden layer:  "+str(self.hiddenLayer.getDimension())+"\n"

class NeuralNetwork1C1P3LReLU(NeuralNetwork3Layers) :
    def __init__(self,
        inputDim,
        convFilterNum, convFilterDim, convImageSize,
        poolFilterDim, poolImageSize,
        hiddenDim,
        outputDim,
        seed = 1):

        #畳み込み+プーリング付き３層ニューラルネット
        self.inputLayer = layer.InputLayer(inputDim)
        self.convLayer = layer.ConvolutionalLayer(
                    convFilterNum,convFilterDim,convImageSize,
                    self.inputLayer,seed)
        self.poolLayer = layer.PoolingLayer(
                    poolFilterDim,poolImageSize,
                    self.convLayer)
        self.hiddenLayer = layer.HiddenLayer(hiddenDim,self.poolLayer,seed)
        self.outputLayer = layer.OutputLayer(outputDim,self.hiddenLayer,seed)

        self.hiddenLayer.setActivator(util.relu)
        self.hiddenLayer.setBackPropagator(learning.backPropReLU)

        # 正解データ
        self.answer = np.zeros(outputDim)

    def getInfo(self) :
        return "\nNeuralNetwork1C1P3LReLU" + "\nconvolution layer: filter num "+str(self.convLayer.filterNum) + "\n                   filter dim "+str(self.convLayer.filterDim) + "\npooling layer    : filter dim "+str(self.poolLayer.filterDim) + "\nhidden layer     : dim "+str(self.hiddenLayer.getDimension())+"\n"

    def save(self,
        fileCW = "./learned_data/convWeight",
        fileCS = "./learned_data/convShift",
        fileHW = "./learned_data/hiddenWeight",
        fileHS = "./learned_data/hiddenShift",
        fileOW = "./learned_data/outputWeight",
        fileOS = "./learned_data/outputShift") :

        np.save(fileCW, self.convLayer.getWeight())
        np.save(fileCS, self.convLayer.getShift())
        np.save(fileHW, self.hiddenLayer.getWeight())
        np.save(fileHS, self.hiddenLayer.getShift())
        np.save(fileOW, self.outputLayer.getWeight())
        np.save(fileOS, self.outputLayer.getShift())
        print("saved.")

    def load(self,
        fileCW = "./learned_data/convWeight.npy",
        fileCS = "./learned_data/convShift.npy",
        fileHW = "./learned_data/hiddenWeight.npy",
        fileHS = "./learned_data/hiddenShift.npy",
        fileOW = "./learned_data/outputWeight.npy",
        fileOS = "./learned_data/outputShift.npy") :

        self.convLayer.setWeight(np.load(fileCW))
        self.convLayer.setShift(np.load(fileCS))
        self.hiddenLayer.setWeight(np.load(fileHW))
        self.hiddenLayer.setShift(np.load(fileHS))
        self.outputLayer.setWeight(np.load(fileOW))
        self.outputLayer.setShift(np.load(fileOS))
        print("loaded.")

class NeuralNetwork2C2P3LReLU(NeuralNetwork3Layers):
    def __init__(self,
        inputDim,
        conv1FilterNum, conv1FilterDim, conv1ImageSize,
        pool1FilterDim, pool1ImageSize,
        conv2FilterNum, conv2FilterDim, conv2ImageSize,
        pool2FilterDim, pool2ImageSize,
        hiddenDim,
        outputDim,
        seed = 1):

        #畳み込み+プーリング+畳み込み+プーリング+３層ニューラルネット
        self.inputLayer = layer.InputLayer(inputDim)
        self.convLayer1 = layer.ConvolutionalLayer(
                    conv1FilterNum,conv1FilterDim,conv1ImageSize,
                    self.inputLayer,seed)
        self.poolLayer1 = layer.PoolingLayer(
                    pool1FilterDim,pool1pool1ImageSize,
                    self.convLayer1)
        self.convLayer2 = layer.ConvolutionalLayer(
                    conv2FilterNum,conv2FilterDim,conv2conv2ImageSize,
                    self.poolLayer1,seed)
        self.poolLayer2 = layer.PoolingLayer(
                    pool2FilterDim,pool2pool2ImageSize,
                    self.convLayer2)
        self.hiddenLayer = layer.HiddenLayer(hiddenDim,self.poolLayer2,seed)
        self.outputLayer = layer.OutputLayer(outputDim,self.hiddenLayer,seed)

        self.hiddenLayer.setActivator(util.relu)
        self.hiddenLayer.setBackPropagator(learning.backPropReLU)


        # 正解データ
        self.answer = np.zeros(outputDim)

    def save(self,
        fileCW1 = "./learned_data/convWeight1",
        fileCS1 = "./learned_data/convShift1",
        fileCW2 = "./learned_data/convWeight2",
        fileCS2 = "./learned_data/convShift2",
        fileHW = "./learned_data/hiddenWeight",
        fileHS = "./learned_data/hiddenShift",
        fileOW = "./learned_data/outputWeight",
        fileOS = "./learned_data/outputShift") :

        np.save(fileCW1, self.convLayer1.getWeight())
        np.save(fileCS1, self.convLayer1.getShift())
        np.save(fileCW2, self.convLayer2.getWeight())
        np.save(fileCS2, self.convLayer2.getShift())
        np.save(fileHW, self.hiddenLayer.getWeight())
        np.save(fileHS, self.hiddenLayer.getShift())
        np.save(fileOW, self.outputLayer.getWeight())
        np.save(fileOS, self.outputLayer.getShift())
        print("saved.")

    def load(self,
        fileCW1 = "./learned_data/convWeight1.npy",
        fileCS1 = "./learned_data/convShift1.npy",
        fileCW2 = "./learned_data/convWeight2.npy",
        fileCS2 = "./learned_data/convShift2.npy",
        fileHW = "./learned_data/hiddenWeight.npy",
        fileHS = "./learned_data/hiddenShift.npy",
        fileOW = "./learned_data/outputWeight.npy",
        fileOS = "./learned_data/outputShift.npy") :

        self.convLayer1.setWeight(np.load(fileCW1))
        self.convLayer1.setShift(np.load(fileCS1))
        self.convLayer2.setWeight(np.load(fileCW2))
        self.convLayer2.setShift(np.load(fileCS2))
        self.hiddenLayer.setWeight(np.load(fileHW))
        self.hiddenLayer.setShift(np.load(fileHS))
        self.outputLayer.setWeight(np.load(fileOW))
        self.outputLayer.setShift(np.load(fileOS))
        print("loaded.")

    def getInfo(self) :
        "\nNeuralNetwork2C2P3LReLU\n"
