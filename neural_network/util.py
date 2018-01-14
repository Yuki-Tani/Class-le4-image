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

### activator ###

def identity(x):
    return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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
