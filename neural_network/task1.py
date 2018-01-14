import ioex
import layer
import util

print("### task1 ###")

# 入出力準備
inputM = ioex.InputManager()
outputM = ioex.OutputManager()
testingData = inputM.getMnistTestingData()

# ３層ニューラルネットワーク
inputLayer = layer.InputLayer(28*28)
hiddenLayer = layer.HiddenLayer(50,inputLayer,1,1)
outputLayer = layer.OutputLayer(10,hiddenLayer,1,1)

# ニューラルネットワークの設定
#outputLayer.setActivator(util.correctedSoftmax)

#入出力
loop = True
while(loop):
    # 入力
    targetNum = inputM.selectNumber()
    sample = testingData.getSingleData(targetNum)
    inputLayer.setInput(sample.getImage())

    # 出力
    result = outputLayer.calculate()
    outputM.printMaxLikelihood(result)
    print(result)

    print("\ncontinue?")
    loop = inputM.selectYesOrNo()

print("Bye.")
