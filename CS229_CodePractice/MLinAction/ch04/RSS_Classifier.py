import feedparser
import numpy as np

import bayes


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# bag of words model
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []

    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = bayes.trainNB1(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(feed1, feed0):
    vocabList, p0V, p1V = localWords(feed1, feed0)
    top1 = []
    top0 = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            top0.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            top1.append((vocabList[i], p1V[i]))

    sorted0 = sorted(top0, key=lambda pair: pair[1], reverse=True)
    print('OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**OS**')
    for item in sorted0:
        print(item[0])
    sorted1 = sorted(top1, key=lambda pair: pair[1], reverse=True)
    print("Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam**Yam")
    for item in sorted1:
        print(item[0])


if __name__ == '__main__':
    oschina = feedparser.parse('https://www.espn.com/espn/rss/news')
    yam = feedparser.parse('https://www.espn.com/espn/rss/nba/news')
    for i in range(10):
        vocabList, pOS, pYam = localWords(oschina, yam)
    # getTopWords(oschina, yam)
