import random

import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    # return 1.0 / (1 + np.exp(-inX))
    # 对sigmoid函数的优化，避免了出现极大的数据溢出
    if inX >= 0:
        return 1.0 / (1 + np.exp(-inX))
    else:
        return np.exp(inX) / (1 + np.exp(inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    m, n = np.shape(dataMatrix)
    alpha = 1e-3
    maxCycles = 500
    weights = np.ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        # dl/dw = X.T * (y - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # decision boundary, 0 = w0 + w1*x1 + w2*x2 (x2 is y)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        featureArr = []
        for i in range(21):
            featureArr.append(float(curLine[i]))
        trainingSet.append(featureArr)
        trainingLabels.append(float(curLine[21]))

    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        curLine = line.strip().split('\t')
        featureArr = []
        for i in range(21):
            featureArr.append(float(curLine[i]))
        if int(classifyVector(np.array(featureArr), trainWeights)) != int(curLine[21]):
            errorCount += 1

    errorRate = float(errorCount) / numTestVec
    print('The error rate of this test is %f' % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('After %d iterations the average error rate is: %f' % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    multiTest()
