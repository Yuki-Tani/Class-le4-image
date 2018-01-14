import ioex
import layer
import util
import nn
import sys
import matplotlib.pyplot as plt
from pylab import cm

print("### task4 ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
testingData = inputM.getMnistTestingData()

neuralNet = nn.NeuralNetwork3Layers(28*28, 50, 10)

print("---- load section ----")
neuralNet.load()

print(" ---- mode select ----")
print("0 : single select mode")
print("1 : percent check mode")

mode = inputM.selectNumber()

if mode == 0 :
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

elif mode == 1:
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


print("Bye.")
