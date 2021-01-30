from kMeans import *


def biKMeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # initial one cluster
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2

    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClusAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClusAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit and not Split: ', sseSplit, sseNotSplit)

            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClusAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # update cluster index
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # update new centroid coordinate
        print('new coordinate: ', bestNewCents)
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return mat(centList), clusterAssment


if __name__ == '__main__':
    dataMat = mat(loadDataSet('testSet2.txt'))
    centList, clusterAssment = biKMeans(dataMat, 3)
    print(centList)
