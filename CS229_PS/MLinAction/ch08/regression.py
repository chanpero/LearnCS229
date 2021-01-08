from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        featArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            featArr.append(float(curLine[i]))
        dataMat.append(featArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = matmul(xMat.T, xMat)
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def plotLinearRegres(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    print(xCopy.shape, yHat.shape)
    ax.plot(xCopy[:, 1], yHat, 'g--')
    plt.show()


if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    plotLwlr(xArr, xArr, yArr, yHat)
