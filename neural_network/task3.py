import ioex
import layer
import util
import nn

print("### task3 ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
trainingData = inputM.getMnistTrainingData()

neuralNet = nn.NeuralNetwork3Layers(28*28, 50, 10)

print("---- load section ----")
print("use weight and shift data file?")
useFile = inputM.selectYesOrNo()
if useFile :
    neuralNet.load()

print("---- learning section ----")
print("input batch size")
batchSize = inputM.selectNumber()

epoch = int(trainingData.images.shape[0] / batchSize)

print("input repeat epoch")
repeatEpoch = inputM.selectNumber()

print("input seed")
seed = inputM.selectNumber()

print("input update ratio (mili)")
updateRatio = inputM.selectNumber() * 0.001

#test
#repeatEpoch = 1
#epoch = 1

for e in range(0,repeatEpoch) :
    print("---- epoch "+str(e+1)+" ----")
    total = 0
    for i in range(0,epoch) :
        neuralNet.learn(trainingData, batchSize, seed, updateRatio)
        print(neuralNet.getLoss(), end = " ")
        percentOfCurrent = neuralNet.getPersentageOfCurrent()
        print(str(percentOfCurrent) + "% (" + str((i*batchSize))  +")")
        total = total + percentOfCurrent / epoch
    print("total percent of current answer : " + str(total) + "%")

print("finish.")


print("---- save section ----")
print("save?")
save = inputM.selectYesOrNo()

if save :
    neuralNet.save()

print("Bye.")
