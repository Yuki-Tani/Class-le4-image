import numpy as np

class ComplexMaker:
    def __init__(self):
        self.seed = 1

    def setSeed(self, seed):
        self.seed = seed

    def getIdentityComplex(self, size):
        matrix = np.zeros(size)
        i = 0
        for r in range(0,size[0]) :
            if(i >= size[1]):
                break
            else:
                matrix[r][i] = 1
                i += 1
        return matrix

    def getRandomComplex(self, size, scale):
        np.random.seed(self.seed)
        # 正規分布
        matrix = np.random.normal(0, scale, size)
        # print("loc:0 scale:"+str(1.0/size[1]))
        return matrix

class BatchMaker:
    def __init__(self, mnistDataBox, batchSize):
        self.box = mnistDataBox
        self.batchSize = batchSize
        self.index = 0

    def reset(self, seed = 0):
        self.index = seed

    def getNextBatch(self):
        imageBatch = self.box.getImageBatch(self.batchSize,self.index)
        answerBatch = self.box.getAnswerVecotrBatch(self.batchSize,self.index)
        self.index = self.index + self.batchSize
        return (imageBatch, answerBatch)

### activator ###

def identity(x):
    return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    x[x<0] = 0
    return x

def softmax(x):
    #print(x)
    shifted = x - x.max()
    #print(shifted)
    numer = np.exp(shifted)
    #print(numer)
    denom = numer.sum(axis=0)
    #print(denom)
    #print(numer/denom)
    return numer / denom

def correctedSoftmax(x):
    corrected = x * 100
    return softmax(corrected)

def crossEntropy(x,target):
    #print(target)
    #print(np.log(x))
    #print((target) * (np.log(x)))
    #print(-(target * np.log(x)).sum(axis=0))
    return np.mean(-(target * np.log(x)).sum(axis=0))

### connector ###

def fullyConnect(inputVector, weight, shift):
    #print( (weight.dot(inputVector) + shift) )
    return weight.dot(inputVector) + shift
