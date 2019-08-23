# use this file to learn naive-bayes classifier
# Expected: generate nbmodel.txt

import sys
import glob
import os
import re
from collections import Counter, defaultdict
import json


def processFold(trainingPaths):
    negDecDoc, negDecWords, negTruDoc, negTruWords = [], [], [], []
    posDecDoc, posDecWords, posTruDoc, posTruWords = [], [], [], []

    for classPath in trainingPaths:
        for path in trainingPaths[classPath]:
            with open(path) as file:
                tmp = tokenization(file.read().strip())
                if "positive" in classPath and "truthful" in classPath:
                    posTruDoc.append(tmp)
                    posTruWords += tmp

                if "positive" in classPath and "deceptive" in classPath:
                    posDecDoc.append(tmp)
                    posDecWords += tmp

                if "negative" in classPath and "truthful" in classPath:
                    negTruDoc.append(tmp)
                    negTruWords += tmp

                if "negative" in classPath and "deceptive" in classPath:
                    negDecDoc.append(tmp)
                    negDecWords += tmp

    return negDecDoc, negDecWords, negTruDoc, negTruWords, posDecDoc, posDecWords, posTruDoc, posTruWords


def tokenization(stringtmp):
    stopwords = ['I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    tmp = re.split('[\s\n]+', stringtmp)
    tmp[0] = tmp[0].lower()
    for i in range(len(tmp)):
        if '.' in tmp[i]:
            tmp[i] = tmp[i].replace('\.', '')
            if i < len(tmp) - 1 and tmp[i + 1] != "I":
                tmp[i + 1] = tmp[i + 1].lower()
        if re.match('\w*\W\w*', tmp[i]) is not None:
            tmp[i] = re.sub('\W', '', tmp[i])
        if tmp[i] is not tmp[i].istitle() and tmp[i] != "I":
            tmp[i] = tmp[i].lower()
        if re.match('([a-zA-Z]+?)(s|es)($)', tmp[i]) and tmp[i] not in stopwords:
            tmp[i] = re.sub('(s|es)', '', tmp[i])
        if tmp[i] == "i":
            tmp[i] = 'I'

    addStopWords = [var[:var.index("\'")] + var[var.index("\'") + 1:] for var in stopwords if "\'" in var]

    stopwords += addStopWords

    tmpReal = []

    for var in tmp:
        if var not in stopwords:
            tmpReal.append(var)

    return tmpReal


def getVocabulary(allText):
    vocabulary = Counter(allText)
    if '' in vocabulary:
        vocabulary.pop('')
    return vocabulary


def trainNaiveBayesMultinomial(documents, vocabulary, Words, classDoc):
    noDoc = float(sum([len(list) for list in documents.values()]))
    prior = {}
    condProb = {}
    for var in classDoc:
        varProb = {}
        noDocInClass = float(len(documents[var]))
        prior[var] = float(noDocInClass / noDoc)

        myCounter = Counter(Words[var])
        for word in vocabulary.keys():
            if word in myCounter:
                varProb[word] = float((myCounter[word] + 1)) / float((sum(myCounter.values()) + len(vocabulary.keys())))
            else:
                varProb[word] = 1.0 / float((sum(myCounter.values()) + len(vocabulary.keys())))

        condProb[var] = varProb

    return prior, condProb


def start(model_file):
    documentsNegPos = {}
    negative = []
    positive = []

    documentsDecTru = {}
    deceptive = []
    truthful = []

    WordsPosNeg = {}
    negativeWords = []
    positiveWords = []

    WordsDecTru = {}
    deceptiveWords = []
    truthfulWords = []

    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    trainingPaths = defaultdict(list)
    
    for f in all_files:
        class1, class2, fold, fname = f.split('/')[-4:]
        trainingPaths[class1 + class2].append(f)

    negDecDoc, negDecWords, negTruDoc, negTruWords, posDecDoc, posDecWords, posTruDoc, posTruWords = processFold(
        trainingPaths)

    allText = []

    allText += negDecWords + negTruWords + posDecWords + posTruWords

    negativeWords += negDecWords + negTruWords
    positiveWords += posDecWords + posTruWords
    deceptiveWords += negDecWords + posDecWords
    truthfulWords += negTruWords + posTruWords

    positive += posDecDoc + posTruDoc
    negative += negDecDoc + negTruDoc
    deceptive += negDecDoc + posDecDoc
    truthful += negTruDoc + posTruDoc

    documentsNegPos['positive'] = positive
    documentsNegPos['negative'] = negative
    documentsDecTru['truthful'] = truthful
    documentsDecTru['deceptive'] = deceptive

    WordsPosNeg['positive'] = positiveWords
    WordsPosNeg['negative'] = negativeWords
    WordsDecTru['truthful'] = truthfulWords
    WordsDecTru['deceptive'] = deceptiveWords

    classDocPosNeg = ["positive", "negative"]
    classDocTruDec = ["truthful", "deceptive"]

    vocabulary = getVocabulary(allText)

    priorTruDec, condProbTruDec = trainNaiveBayesMultinomial(documentsDecTru, vocabulary, WordsDecTru, classDocTruDec)
    priorPosNeg, condProbPosNeg = trainNaiveBayesMultinomial(documentsNegPos, vocabulary, WordsPosNeg, classDocPosNeg)


    with open(model_file, 'w+') as file:
        tmp = [vocabulary, classDocTruDec, priorTruDec, condProbTruDec, classDocPosNeg, priorPosNeg,condProbPosNeg]
        file.write(json.dumps(tmp))


if __name__ == "__main__":
    model_file = "nbmodel.txt"
    start(model_file)
