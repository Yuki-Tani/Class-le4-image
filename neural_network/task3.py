import ioex
import layer
import util
import nn
import sys

SEED = 100

print("### task3 ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
trainingData = inputM.getMnistTrainingData()

neuralNet = nn.NeuralNetwork3Layers(28*28, 50, 10, SEED)

print("---- load section ----")
print("use weight and shift data file?")
useFile = inputM.selectYesOrNo()
if useFile :
    neuralNet.load()

print("---- learning section ----")
print("input batch size")
batchSize = inputM.selectNumber()

batchMaker = util.BatchMaker(trainingData,batchSize)
epochSize = int(trainingData.images.shape[0] / batchSize)

print("input repeat epoch")
repeatEpoch = inputM.selectNumber()

print("input seed")
seed = inputM.selectNumber()

print("input update ratio (mili)")
updateRatio = inputM.selectNumber() * 0.001

#test
#repeatEpoch = 1
#epochSize = 1

#計測スタート
import time
start = time.time()

for e in range(0,repeatEpoch) :
    print("---- epoch "+str(e+1)+" ----")
    batchMaker.reset(seed + e)
    total = 0
    totalLoss = 0
    for i in range(0,epochSize) :
        #学習
        (imageBatch,answerBatch) = batchMaker.getNextBatch()
        neuralNet.learn(imageBatch,answerBatch, updateRatio)
        loss = neuralNet.getLoss()
        percentOfCurrent = neuralNet.getPersentageOfCurrent()
        #表示
        sys.stdout.write("\r%d" % ((i+1)*batchSize))
        sys.stdout.flush()
        #print(loss, end = " ")
        #print(str(percentOfCurrent) + "% (" + str((i*batchSize))  +")")
        #更新
        total = total + percentOfCurrent / epochSize
        totalLoss = totalLoss + loss / epochSize
    print("\ntotal loss : " + str(totalLoss))
    print("total percent of current answer : " + str(total) + "%")

print("finish.")
#計測終了
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print("---- save section ----")
print("save?")
save = inputM.selectYesOrNo()

if save :
    neuralNet.save()

print("Bye.")
