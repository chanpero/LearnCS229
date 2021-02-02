from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDimDataMat = meanRemoved * redEigVects
    reconMat = (lowDimDataMat * redEigVects.T) + meanVals
    return lowDimDataMat, reconMat


def replaceNanWithMean():
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDimMat, reconMat = pca(dataMat, 1)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

