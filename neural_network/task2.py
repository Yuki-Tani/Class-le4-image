import ioex
import layer
import util
import nn

BATCH_QUANTITY = 100

print("### task2 ###")

print("batch mode")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
testingData = inputM.getMnistTestingData()
testingData.shuffle()

neuralNet = nn.NeuralNetwork3Layers(28*28, 50, 10)

loop = True
while(loop):
    print("Input random seed number of batch.")
    begin = inputM.selectNumber()
    imageBatch = testingData.getImageBatch(BATCH_QUANTITY, begin)
    answerBatch = testingData.getAnswerVecotrBatch(BATCH_QUANTITY, begin)
    neuralNet.setInputBatch(imageBatch)
    neuralNet.setAnswerBatch(answerBatch)

    print("start")
    result = neuralNet.calculate()
    #print(result)
    totalLoss = neuralNet.getLoss()

    print("\nResult.")
    print("totalLoss : " + str(totalLoss))

    print("\ncontinue?")
    loop = inputM.selectYesOrNo()

print("Bye.")

######################################################
"""

print("\n##########\nrepeat mode")
# ３層ニューラルネットワーク
inputLayer = layer.InputLayer(28*28)
hiddenLayer = layer.HiddenLayer(50,inputLayer)
outputLayer = layer.OutputLayer(10,hiddenLayer)

#入出力
loop = True
while(loop):
    # 入力
    print("input seed number.")
    begin = inputM.selectNumber()
    begin = begin % (BATCH_QUANTITY - begin + 1)
    #testingData.randomInit(seed)
    totalLoss = 0

    for i in range(begin,BATCH_QUANTITY + begin):
        sample = testingData.getSingleData(i)#testingData.nextDataRandom()
        target = sample.getAnswerAsVector((10,1))
        #if i==0 :
        #    print(sample.getImage().flatten())
        #    print(target)
        inputLayer.setInput(sample.getImage())
        result = outputLayer.calculate()
        #print(result)
        loss = outputLayer.getLoss(target)
        totalLoss += loss / BATCH_QUANTITY

    print("\nResult.")
    print("totalLoss : " + str(totalLoss))

    print("\ncontinue?")
    loop = inputM.selectYesOrNo()

print("Bye.")
"""
