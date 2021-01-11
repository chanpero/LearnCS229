from numpy import *

import regression


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


# regularize by columns
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # calc mean then subtract it off
    inVar = var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIter=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat = regularize(xMat)
    yMean = mean(yMat, 0)
    yMat = yMat - yMean

    m, n = shape(xMat)
    returnMat = zeros((numIter, n))
    ws = zeros((n, 1))
    wsMax = ws.copy()

    for i in range(numIter):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


if __name__ == '__main__':
    print('Forward Step-wise Regression: ')
    xArr, yArr = regression.loadDataSet('abalone.txt')
    stageWise(xArr, yArr, 0.005, 5000)

    print('Standard Regression: ')
    weights = regression.standRegres(regularize(mat(xArr)), (mat(yArr).T - mean(mat(yArr).T, 0)).T)
    print(weights.T)
