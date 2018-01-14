import numpy as np

class MnistDataBox:
    def __init__(self, images, answers) :
        self.images = images
        self.answers = answers
        self.boxSize = answers.size
        self.order = np.arange(self.boxSize)
        self.index = 0

    def getSingleData(self, num) :
        return MnistData(self.images[num],self.answers[num])

    def randomInit(self, seed = 1):
        self.order = np.arange(self.boxSize)
        np.random.seed(seed)
        np.random.shuffle(self.order)
        self.index = 0

    def nextDataRandom(self):
        data = self.getSingleData(self.order[self.index])
        self.index = (self.index + 1) % self.boxSize
        return data

    def shuffle(self, seed = 1):
        np.random.seed(seed)
        np.random.shuffle(self.images)
        np.random.seed(seed)
        np.random.shuffle(self.answers)

    def getImageBatch(self, batchSize, shift = 0) :
        shift = shift % (batchSize - shift + 1)
        batch = self.images[shift : batchSize + shift]
        return batch.reshape((batchSize,-1)).T

    def getAnswerVecotrBatch(self, batchSize, shift = 0) :
        shift = shift % (batchSize - shift + 1)
        batch = np.zeros((10,batchSize))
        for i in range(0,batchSize) :
            batch[self.answers[i+shift]][i] = 1
        return batch

    def getSize(self):
        return self.boxSize

class MnistData:
    def __init__(self, image, answer) :
        self.image = image
        self.answer = int(answer)

    def getSize(self):
        return self.image.size

    def getImage(self):
        return self.image

    def getAnswer(self):
        return self.answer

    def getAnswerAsVector(self, size = None):
        ans = np.zeros(10)
        print(self.answer)
        ans[self.answer] = 1
        if size is None :
            return ans
        else :
            return ans.reshape(size)
