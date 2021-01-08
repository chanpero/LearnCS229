from regression import *


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k = 0.1: ', rssError(abY[0:99], yHat01.T))
    print('k = 1  : ', rssError(abY[0:99], yHat1.T))
    print('k = 10 : ', rssError(abY[0:99], yHat10.T))
    print('---------------------------------')

    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k = 0.1: ', rssError(abY[100:199], yHat01.T))
    print('k = 1  : ', rssError(abY[100:199], yHat1.T))
    print('k = 10 : ', rssError(abY[100:199], yHat10.T))
    print('---------------------------------')

    ws = standRegres(abX[0:99], abY[0:99])
    yHat = mat(abX[100:199]) * ws
    print('Standard Regression: ', rssError(abY[100:199], yHat.T.A))
