import numpy as np
import util
import learning

##################################################
class Layer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.output = np.zeros((dimension, 1))
        #誤差関数
        self.lossFunction = util.crossEntropy

    def calculate(self):
        pass

    def update(self, dEdOut, learningRatio = 0.01):
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
        self.updator = learning.updateBySGD

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
         self.prevLayer.update(dEdIn, learningRatio)

    def setWeight(self,weight):
        if(weight.shape == self.weightSize):
            self.weight = weight
        else:
            print("WARNING! This weight is NOT match to matrix shape.")

    def getWeight(self):
        return self.weight

    def setShift(self,shift):
        #print("shift [layer,112]")
        #print(shift)
        if(shift.shape == self.shiftSize):
            self.shift = shift
        else:
            print("WARNING! This shift is NOT match to matrix shape.")

    def getShift(self):
        return self.shift

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
        self.prevLayer.update(dEdIn, learningRatio)

    def confirmParameters(self):
        print("[Output Layer]")
        print("dimension : "+str(self.dimension))
        print("weight : ")
        print(self.weight)
        print("shift : ")
        print(self.shift)

##################################################
class ConvolutionalLayer(ConversionLayer) :
    def __init__(self, filterNum, filterDim, prevLayer,
                 weightSeed = 1, shiftSeed = 1) :
        # dimension = x^2 * f
        super().__init__(prevLayer.getDimension() * filterNum, prevLayer)
        x = int(np.sqrt(prevLayer.getDimension()))
        self.imageSize = (x,x)
        self.filterNum = filterNum
        self.filterDim = filterDim

        self.preLayer = InternalLayerPreConv(self, self.prevLayer)
        self.internalLayer = HiddenLayer(self.filterNum, self.preLayer,
                                                    weightSeed,shiftSeed)
        self.postLayer = InternalLayerPostConv(self, self.internalLayer)

        self.internalLayer.setActivator(util.relu)
        self.internalLayer.setBackPropagator(learning.backPropReLU)

    def setImageSize(self, size):
        # (H,W)
        self.imageSize = size

    def calculate(self):
        self.output = self.postLayer.calculate()
        return self.output

    def update(self, dEdOut, learningRatio = 0.01):
        self.postLayer.update(dEdOut,learningRatio)

    def confirmParameters(self):
        print("[Convolutional Layer]")
        print("filter num : "+str(self.filterNum))
        print("filter dim : "+str(self.filterDim))
        print("dimension : "+str(self.dimension))
        print("weight :")
        print(self.weight)
        print("shift :")
        print(self.shift)

    def setWeight(self,weight):
        self.internalLayer.setWeight(weight)
    def setShift(self,shift):
        self.internalLayer.setShift(shift)
    def setActivator(self,function):
        self.internalLayer.setActivator(function)
    def setConnector(self,function):
        self.internalLayer.setConnector(function)
    def setBackPropagator(self, function) :
        self.internalLayer.setBackPropagator(function)
    def setUpdator(self, function) :
        self.internalLayer.setUpdator(function)
    def getWeight(self):
        return self.internalLayer.weight
    def getShift(self):
        return self.internalLayer.shift

class InternalLayerPreConv(Layer) :
    def __init__(self, convLayer, prevLayer):
        # dimension = filterDim^2
        super().__init__(convLayer.filterDim**2)
        self.convLayer = convLayer
        self.prevLayer = prevLayer

    def calculate(self) :
        inputMatrix = self.prevLayer.calculate()
        batchSize = inputMatrix.shape[1]
        imageSize = self.convLayer.imageSize
        filterDim = self.convLayer.filterDim
        #(H*W) * B -> H * W * B
        mat = inputMatrix.reshape(
                imageSize[0],imageSize[1],batchSize)
        # padding (H+d-1) * (W+d-1) * B
        # スライスのため+1でパディング
        pw = filterDim // 2
        mat = np.pad(mat,((pw,pw+1),(pw,pw+1),(0,0)),'constant')

        newInputMat = np.empty([0,(mat.shape[0]-(filterDim-1)-1)*
                        (mat.shape[1]-(filterDim-1)-1)*batchSize])
        for i in range(0,filterDim):
            for j in range(0,filterDim):
                # H * W * B
                sl = mat[i:i-(filterDim-1)-1,j:j-(filterDim-1)-1]
                # B * H * W
                sl = sl.transpose(2,0,1)
                # 1 * (B*H*W)
                sl = sl.reshape(1,-1)
                newInputMat = np.concatenate((newInputMat,sl),axis = 0)
        self.output = newInputMat
        #print("pre culc 275")
        #print(self.output)
        return self.output

    def update(self, dEdOut, learningRatio = 0.01) :
        #print("pre up 280")
        #print(dEdOut)
        wh = self.convLayer.imageSize[0]*self.convLayer.imageSize[1]
        mat = dEdOut[(self.convLayer.filterDim**2) // 2]
        dEdIn = mat.reshape(-1,wh).T
        #print(dEdIn)
        self.prevLayer.update(dEdIn, learningRatio)

class InternalLayerPostConv(Layer) :
    def __init__(self, convLayer, prevLayer):
        super().__init__(convLayer.dimension)
        self.convLayer = convLayer
        self.prevLayer = prevLayer

    def calculate(self) :
        outputMatrix = self.prevLayer.calculate()
        f = self.convLayer.filterNum
        wh = self.convLayer.imageSize[0]*self.convLayer.imageSize[1]
        # f * (B * W * H) -> f * B * (W*H)
        mat = outputMatrix.reshape(f,-1,wh)
        # B * f * (W*H)
        mat = mat.transpose(1,0,2)
        # B * (f * W * H) -> (f * W * H) * B
        self.output = mat.reshape(-1, f*wh).T
        #print("post culc 304")
        #print(self.output)
        return self.output

    def update(self, dEdOut, learningRatio = 0.01):
        #print("post up 309")
        #print(dEdOut)
        f = self.convLayer.filterNum
        wh = self.convLayer.imageSize[0]*self.convLayer.imageSize[1]
        # (f * W * H) * B -> B * f * (W*H)
        mat = dEdOut.T.reshape(-1,f,wh)
        # B * f * (W*H) -> f * B * (W*H)
        mat = mat.transpose(1,0,2)
        # f * B * (W*H) -> f * (B*W*H)
        dEdIn = mat.reshape(f,-1)
        #print(dEdIn)
        self.prevLayer.update(dEdIn, learningRatio)

class PoolingLayer(ConversionLayer) :
    def __init__(self, filterDim, prevLayer) :
        # dimension = (f * x^2) / pooling^2
        super().__init__(prevLayer.getDimension() // filterDim**2 , prevLayer)
        x = int(np.sqrt(prevLayer.getDimension()/(filterDim**2)))
        self.imageSize = (x*(filterDim**2),x)
        self.filterDim = filterDim
        self.arrangedMat = None
        self.maxedMat = None

    def setImageSize(self, size) :
        #(H,W)
        self.imageSize = size

    def calculate(self) :
        mat = self.prevLayer.calculate().T
        pd = self.filterDim
        h = self.imageSize[0]
        w = self.imageSize[1]

        mat = mat.reshape((-1, h//pd, pd, w//pd, pd))
        mat = mat.transpose((0,1,3,2,4))
        self.arrangedMat = mat.reshape(-1, h//pd, w//pd, pd**2)
        self.maxedMat = self.arrangedMat.max(axis = 3)
        self.output = self.maxedMat.reshape(-1, h*w//(pd**2)).T
        return self.output

    def update(self, dEdOut, learningRatio = 0.01) :
        h = self.imageSize[0]
        w = self.imageSize[1]
        pd = self.filterDim
        # B * h//pd * w//pd * pd**2 -> pd**2 * B * h//pd * w//pd
        mat = self.arrangedMat
        mat = mat.transpose(3,0,1,2)
        # B * h//pd * w//pd
        maxed = self.maxedMat
        # dEdOutの変形
        # (h*w//(pd**2)) * B -> B * h//pd, w//pd
        eff = dEdOut.T.reshape(-1,h//pd,w//pd)

        # pd**2 * B * h//pd * w//pd
        base = np.zeros(mat.shape)
        base[mat==maxed] = 1
        dEdIn = base*eff

        #dEdInの変形
        #(pd**2)*B*(h//pd)*(w//pd) -> pd * pd * B*(h//pd)*(w//pd)
        dEdIn = dEdIn.reshape(pd,pd,-1,h//pd,w//pd)
        #pd * pd * B*(h//pd)*(w//pd) -> B*(h//pd)*pd*(w//pd)*pd
        dEdIn = dEdIn.transpose(2,3,0,4,1)
        #B*(h//pd)*pd*(w//pd)*pd -> B * (h*w)
        dEdIn = dEdIn.reshape(-1,h*w)
        #print("pool up 400")
        #print(dEdIn)
        self.prevLayer.update(dEdIn.T, learningRatio)
