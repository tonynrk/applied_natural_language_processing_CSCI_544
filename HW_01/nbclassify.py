# use this file to classify using naive-bayes classifier
# Expected: generate nboutput.txt

import sys
import glob
import os
import math
import re
import json

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



def applyMultinomialNB(classDoc, vocabulary, prior, condProb, document):
    w = ''
    with open(document) as file:
        w = file.read().rstrip()

    words = tokenization(w)

    words = [word for word in words if word in vocabulary]
    score = {}
    for c in classDoc:
        score[c] = math.log(prior[c])
        for word in words:
            score[c] += math.log(condProb[c][word])

    return max(score, key=score.get)


def testing(testingPaths, vocabulary, classDocTruDec, priorTruDec, condProbTruDec, classDocPosNeg, priorPosNeg,
            condProbPosNeg):
    tmp = []
    for path in testingPaths:
        classTruDec = applyMultinomialNB(classDocTruDec, vocabulary, priorTruDec, condProbTruDec, path)
        classPosNeg = applyMultinomialNB(classDocPosNeg, vocabulary, priorPosNeg, condProbPosNeg, path)
        tmp.append(classTruDec + " " + classPosNeg + " " + path + "\n")

    return tmp


def main(output_file):

    [vocabulary, classDocTruDec, priorTruDec, condProbTruDec, classDocPosNeg, priorPosNeg,
     condProbPosNeg] = json.loads(open('nbmodel.txt').read())

    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))


    result = testing(all_files, vocabulary, classDocTruDec, priorTruDec, condProbTruDec, classDocPosNeg, priorPosNeg,
                     condProbPosNeg)

    with open(output_file, 'w+') as file:
        for sentence in result:
            file.write(sentence)

    return


if __name__ == "__main__":
    output_file = "nboutput.txt"
    main(output_file)