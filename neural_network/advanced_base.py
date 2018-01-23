import ioex
import layer
import util
import nn
import sys
import matplotlib.pyplot as plt
from pylab import cm

DEFAULT_TARGET_FILE_NAME = "neural_network/learned_data/answerForContest.txt"
LOG_FILE_NAME = "log/learning_log.txt"

def singleCheck(neuralNet, mnistDataBox, num, inputM, allShowPic = False) :
    mnist = mnistDataBox.getSingleData(num)
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
    showPic = True
    if not allShowPic :
        print("\nshow picture?")
        showPic = inputM.selectYesOrNo()
    if showPic :
        plt.imshow(image, cmap=cm.gray)
        plt.show()

def percentCheck(neuralNet, mnistDataBox, repeat, title = "") :
    currentQuantity = 0
    print("calculate")
    for i in range(0,repeat):
        mnist = mnistDataBox.getSingleData(i)
        image = mnist.getImage()
        answer = mnist.getAnswer()
        neuralNet.setInput(image)
        result = neuralNet.calculate().argmax()
        if result == answer:
            currentQuantity = currentQuantity + 1
        #表示
        sys.stdout.write("\r%d" % (i+1))
        sys.stdout.flush()

    percent = currentQuantity / repeat * 100
    print("\n" + title + " Result")
    print("percent of current : " + str(percent) + "%")
    return percent

def mistakeCheck(neuralNet, mnistDataBox, num, repeat, inputM) :
    for i in range(num, num+repeat):
        mnist = mnistDataBox.getSingleData(i)
        image = mnist.getImage()
        answer = mnist.getAnswer()
        neuralNet.setInput(image)
        result = neuralNet.calculate().argmax()
        if result != answer:
            singleCheck(neuralNet, mnistDataBox, i, inputM, True)
            print("\nexit?")
            if inputM.selectYesOrNo() :
                return

def makeAnswerFile(neuralNet, mnistDataBox, outputManager,
                    targetFileName = DEFAULT_TARGET_FILE_NAME) :
    print("\n---- calculate ----")
    text = ""
    for i in range(0,mnistDataBox.getSize()):
        mnist = mnistDataBox.getSingleData(i)
        image = mnist.getImage()
        neuralNet.setInput(image)
        result = neuralNet.calculate().argmax()
        text = text + str(result)
        if i != mnistDataBox.getSize() - 1 :
            text = text + "\n"
        #表示
        sys.stdout.write("\r%d" % (i+1))
        sys.stdout.flush()

    print("\n---- make file ----")
    outputManager.printTextFile(targetFileName,text)
    print("done.")

###################################################################

def runLearning(neuralNet, inputM, outputM, isCompleteMode = False):
    trainingData = inputM.getMnistTrainingData()

    if isCompleteMode :
        testingData = inputM.getMnistTestingData()
        contestData = inputM.getLe4MnistData()

    print("---- load section ----")
    print("use weight and shift data file?")
    useFile = inputM.selectYesOrNo()
    if useFile :
        neuralNet.load()

    #入力
    print("input batch size")
    batchSize = inputM.selectNumber()
    print("input repeat epoch")
    repeatEpoch = inputM.selectNumber()
    print("input seed")
    seed = inputM.selectNumber()
    print("input update ratio (mili)")
    updateRatio = inputM.selectNumber() * 0.001

    batchMaker = util.BatchMaker(trainingData,batchSize)
    epochSize = int(trainingData.images.shape[0] / batchSize)

    print("---- learning section ----")

    if isCompleteMode :
        log = outputM.addPrintTextFile(LOG_FILE_NAME,neuralNet.getInfo())


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

        #エポック終了時の表示
        print("\ntotal loss : " + str(totalLoss))
        print("total percent of current answer : " + str(total) + "%")
        if isCompleteMode :
            #他データで正答率確認
            size = testingData.getSize()
            testingPercent = percentCheck(neuralNet,testingData,size,"Testing")
            contestPercent = percentCheck(neuralNet,contestData,200,"Contest")
            log = str(e)+" "+str(totalLoss)+" "+str(total)+" "+str(testingPercent)+" "+str(contestPercent)+"\n"
            outputM.addPrintTextFile(LOG_FILE_NAME, log)


    print("finish.")

    print("---- save section ----")
    print("save?")
    save = inputM.selectYesOrNo()
    if save :
        neuralNet.save()

##################################################

def runMainController(neuralNet, inputM, outputM):
    print(" ---- mode select ----")
    print("0 : [training] learning mode")
    print("10 : [testing] single select mode")
    print("11 : [testing] percent check mode")
    print("12 : [testing] mistake check mode")
    print("20 : [contest] single select mode")
    print("21 : [contest] percent check mode")
    print("22 : [contest] mistake check mode")
    print("23 : [contest] file making mode")
    print("30 : [training-test] complete mode")
    mode = inputM.selectNumber()

    # [training] learning mode
    # [training-test] complete mode
    if mode == 0 or mode == 30 :
        runLearning(neuralNet,inputM,outputM, mode == 30)

    else :
        if mode == 10 or mode == 11 or mode == 12 :
            testingData = inputM.getMnistTestingData()
        if mode == 20 or mode == 21 or mode == 22 or mode == 23 :
            testingData = inputM.getLe4MnistData()

        print("---- load section ----")
        neuralNet.load()

        # [testing] single select mode
        # [contest] single select mode
        if mode == 10 or mode == 20 :
            print("select image number")
            num = inputM.selectNumber()
            loop = True
            while(loop):
                singleCheck(neuralNet,testingData,num, inputM)
                print("\nnext?")
                loop = inputM.selectYesOrNo()
                num = num + 1

        # [testing] percent check mode
        elif mode == 11 :
            repeat = testingData.getSize()
            percentCheck(neuralNet,testingData,repeat,"Testing")

        # [contest] percent check mode
        elif mode == 21 :
            repeat = 200
            percentCheck(neuralNet,testingData,repeat,"Contest")

        # [testing] mistake check mode
        # [contest] mistake check mode
        elif mode == 12 or mode == 22 :
            print("select start image number")
            num = inputM.selectNumber()
            if mode == 12 :
                repeat = testingData.getSize()
            else :
                repeat = 200
            mistakeCheck(neuralNet,testingData,num,repeat,inputM)

        # [contest] file making mode
        elif mode == 23 :
            print("\nwrite answers?")
            if inputM.selectYesOrNo() :
                makeAnswerFile(neuralNet,testingData,outputM)

    print("Bye.")
