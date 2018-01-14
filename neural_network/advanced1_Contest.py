import ioex
import layer
import util
import nn
import sys
import matplotlib.pyplot as plt
from pylab import cm

TARGET_FILE_NAME = "neural_network/learned_data/answerForContest.txt"
SEED = 100

print("### advanced2 [ReLU] ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
neuralNet = nn.NeuralNetwork3Layers(28*28, 50, 10, SEED)

print(" ---- mode select ----")
print("0 : [training] learning mode")
print("10 : [testing] single select mode")
print("11 : [testing] percent check mode")
print("20 : [contest] single select mode")
print("21 : [contest] percent check mode")
print("22 : [contest] file making mode")

mode = inputM.selectNumber()

###########################################
# [training] learning mode
###########################################
if mode == 0:
    trainingData = inputM.getMnistTrainingData()
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
            #更新
            total = total + percentOfCurrent / epochSize
            totalLoss = totalLoss + loss / epochSize
        print("\ntotal loss : " + str(totalLoss))
        print("total percent of current answer : " + str(total) + "%")

    print("finish.")

    print("---- save section ----")
    print("save?")
    save = inputM.selectYesOrNo()
    if save :
        neuralNet.save()

###########################################
# [testing] single select mode
###########################################
elif mode == 10 :
    testingData = inputM.getMnistTestingData()
    print("---- load section ----")
    neuralNet.load()

    print("select image number")
    num = inputM.selectNumber()

    loop = True
    while(loop):

        mnist = testingData.getSingleData(num)
        image = mnist.getImage()
        answerVec = mnist.getAnswerAsVector()
        answer = mnist.getAnswer()
        neuralNet.setInput(image)
        neuralNet.setAnswer(answerVec)

        result = neuralNet.calculate().flatten()
        rank = result.argsort()[::-1]
        totalLoss = neuralNet.getLoss()

        print("\nResult (No." + str(num) +")",end = "\n\n")

        print("recognition : "+ str(rank[0]))
        print("answer : "+ str(answer), end = "\n\n")

        print("likelihood : ")
        for i in range(0,10):
            likelihood = result[rank[i]]
            print(" " + str(rank[i])+" | " + str(likelihood), end = " ")
            for j in range(0,int(likelihood*10)):
                print("#",end = "")
            print()

        print("totalLoss : " + str(totalLoss))
        plt.imshow(image, cmap=cm.gray)
        plt.show()

        print("\nnext?")
        loop = inputM.selectYesOrNo()
        num = num + 1

###########################################
# [testing] percent check mode
###########################################
elif mode == 11:
    testingData = inputM.getMnistTestingData()
    print("---- load section ----")
    neuralNet.load()

    currentQuantity = 0
    repeat = testingData.getSize()
    print("calculate")
    for i in range(0,repeat):
        mnist = testingData.getSingleData(i)
        image = mnist.getImage()
        answer = mnist.getAnswer()
        neuralNet.setInput(image)
        result = neuralNet.calculate().argmax()
        if result == answer:
            currentQuantity = currentQuantity + 1
        #表示
        sys.stdout.write("\r%d" % (i+1))
        sys.stdout.flush()

    print("\nResult")
    print("percent of current : " + str(currentQuantity / repeat * 100) + "%")

###########################################
# [contest] single select mode
###########################################

elif mode == 20 :
    testingData = inputM.getLe4MnistData()
    print("---- load section ----")
    neuralNet.load()

    print("---- check section ----")
    print("select image number")
    num = inputM.selectNumber()
    loop = True
    while(loop):
        mnist = testingData.getSingleData(num)
        image = mnist.getImage()
        answerVec = mnist.getAnswerAsVector()
        answer = mnist.getAnswer()
        neuralNet.setInput(image)
        neuralNet.setAnswer(answerVec)

        result = neuralNet.calculate().flatten()
        rank = result.argsort()[::-1]
        totalLoss = neuralNet.getLoss()

        print("\nResult (No." + str(num) +")",end = "\n\n")

        print("recognition : "+ str(rank[0]))
        print("answer : "+ str(answer), end = "\n\n")

        print("likelihood : ")
        for i in range(0,10):
            likelihood = result[rank[i]]
            print(" " + str(rank[i])+" | " + str(likelihood), end = " ")
            for j in range(0,int(likelihood*10)):
                print("#",end = "")
            print()

        print("totalLoss : " + str(totalLoss))
        plt.imshow(image, cmap=cm.gray)
        plt.show()

        print("\nnext?")
        loop = inputM.selectYesOrNo()
        num = num + 1

###########################################
# [contest] percent check mode
###########################################
elif mode == 21:

    testingData = inputM.getLe4MnistData()

    print("---- load section ----")
    neuralNet.load()
    currentQuantity = 0
    repeat = 100
    print("---- calculate -----")
    for i in range(0,repeat):
        mnist = testingData.getSingleData(i)
        image = mnist.getImage()
        answer = mnist.getAnswer()
        neuralNet.setInput(image)
        result = neuralNet.calculate().argmax()
        if result == answer:
            currentQuantity = currentQuantity + 1
        sys.stdout.write("\r%d" % (i+1))
        sys.stdout.flush()

    print("\nResult")
    print("percent of current : " + str(currentQuantity / repeat * 100) + "%")

###########################################
# [contest] file making mode
###########################################
elif mode == 22:
    testingData = inputM.getLe4MnistData()
    print("---- load section ----")
    neuralNet.load()
    print("---- calculate ----")
    text = ""
    for i in range(0,testingData.getSize()):
        mnist = testingData.getSingleData(i)
        image = mnist.getImage()
        neuralNet.setInput(image)
        result = neuralNet.calculate().argmax()
        text = text + str(result)
        if i != testingData.getSize() - 1 :
            text = text + "\n"
        #表示
        sys.stdout.write("\r%d" % (i+1))
        sys.stdout.flush()

    print("\ntarget file : " + TARGET_FILE_NAME)
    print("write answers?")
    permition = inputM.selectYesOrNo()

    if permition :
        outputM.printTextFile(TARGET_FILE_NAME,text)
        print("done.")

print("Bye.")
