import sys
import glob
import os
import re
import json
from collections import Counter, defaultdict
import numpy as np

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
                 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    tmp = re.split('[\s\n]+',stringtmp)
    tmp[0] = tmp[0].lower()
    for i in range(len(tmp)):
        if '.' in tmp[i]:
            # tmp[i] = tmp[i].replace('\.', '')
            if i < len(tmp) - 1 and tmp[i + 1] != "I":
                tmp[i + 1] = tmp[i + 1].lower()
        if re.match('\w*\W\w*', tmp[i]) is not None:
             tmp[i] = re.sub('\W', '', tmp[i])
        if re.match('\w*[1234567890]+\w*', tmp[i]) is not None:
            tmp[i] = re.sub('[1234567890]+', '', tmp[i])

        if tmp[i] is not tmp[i].istitle() and tmp[i] != "I":
            tmp[i] = tmp[i].lower()
        if re.match('([a-zA-Z]+?)(s|es)($)', tmp[i]) and tmp[i] not in stopwords:
            tmp[i] = re.sub('(s|es)', '', tmp[i])
        if tmp[i] == "i":
            tmp[i] = 'I'


    addStopWords = [var[:var.index("\'")] + var[var.index("\'") + 1:] for var in stopwords if "\'" in var]

    stopwords += addStopWords

#    addProperNoun = []
    
#    proper1 = ["macys", "african", "meridian", "texas", "james", "hancock", "mariott", "canada", "blu", "california", "spanish"]

#    proper2 =  ["fiji", "george", "ian", "sofitel", "victorian", "san", "sam", "craig", "davies", "america", "german", "talbot", "hiltonbrand", "beatles", "chacago", "scott", "hancok", "sean" ]

#    proper3 = ["thurs", "jr", "js", "chihuahua", "mary", "andrew", "odyssey", "shepard", "candace", "knikerbocker", "shulas", "westin", "theater", "amy", "simmons", "patrick", "caribbean", "schrager", "marriott", "einstein", "october", "av", "confad", "intercontinental", "chris", "chinese", "pricelinecom", "thursday", "lisa", "christy", "november", "osco", "bible", "lstation", "sanjay", "ohare", "morgan", "martens", "chicagos", "ihome", "shouldnt", "orbitz", "mahajan", "fl", "a", "an", "the", "his", "her", "him", "my", "your", "yours", "them", "they", "rottweiler", "september", "nordstrom", "affina", "curtis", "starbucks", "omniall", "marcus", "drake", "england", "randolph", "travellodge", "perry", "robert", "wisconsin", "friday", "hermes", "benedict", "american", "meridien", "affnia", "swisshotel"]

#    proper4 = ["rocco", "millenium", "gulli", "equinox", "sep", "europe", "drury", "conrads", "marilyn", "les", "angelas", "hotelscom", "ireland", "hiltons", "apri", "towne", "styx", "monroe", "ponderosa", "talbott", "janice", "swiss", "cheryl", "ambien", "saturdays", "fortney", "anne", "aerosmith", "santa", "sestons", "timerland", "womens", "jimmy", "steven", "magnificant", "erica", "saks", "aveda", "amabassador", "hampton", "gideons", "swissotels", "entwistle", "shedd", "stephanie", "neiman", "jonathan", "belgian", "spaffina", "weve", "hilton", "nordstroms", "houston", "le", "la", "february", "aon", "ipod", "york", "fitzpatrick", "sharon", "july", "splendido", "roberta", "ben", "ballard", "ginos", "sinatra", "manchester", "perla", "mumbai", "betsy", "ethan", "avenueon", "michigan", "stetsons", "cubs", "manhattans", "mercedez", "irena", "tripadvisor", "bloomingdales", "april", "jersey", "frabreez", "atlanta", "an", "thur", "ave", "paula", "chigaco", "ashley", "houlihans", "paul", "smith", "january", "conrad", "italian", "cleveland", "delaware", "angeles", "daniels", "hardrock", "irish", "erie", "eric", "diego", "illinois", "febreeze", "francisco", "northbridge", "martin", "westfield"]

#    addProperNoun = proper1 + proper2 + proper3 + proper4

    
#    stopwords += addProperNoun
    
    tmpReal = []

    for var in tmp:
        if var not in stopwords and var is not '':
            tmpReal.append(var)

    return tmpReal

def getVocabulary(allText):
    vocabulary = Counter(allText)
    return vocabulary

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

def trainingVanilla(documents,maxIter,myvocabulary,myIndexVocabulary,classdocs):

    weight = np.zeros((len(myvocabulary)), dtype=int)
    b = 0
    for _ in range(maxIter):
        for classdoc in classdocs:
            y = classdocs[classdoc]
            for document in documents[classdoc]:
                x = np.zeros((len(myvocabulary)), dtype=int)
                tmp = []
                for word in document:
                    if x[myIndexVocabulary[word]] == 0:
                        x[myIndexVocabulary[word]] += 1
                        tmp.append(myIndexVocabulary[word])
                a = np.array(x).dot(weight) + b
                if y*a <= 0:
                    for i in tmp:
                        weight[i] += y*x[i]
                    b += y
    return weight,b


def trainingAverage(documents,maxIter,myvocabulary,myIndexVocabulary,classdocs):

    weight = np.zeros((len(myvocabulary)), dtype=int)
    u = np.zeros((len(myvocabulary)), dtype=int)
    b = 0
    B = 0
    c = 1
    y = 0
    for _ in range(maxIter):
        for classdoc in classdocs:
            y = classdocs[classdoc]
            for document in documents[classdoc]:
                x = np.zeros((len(myvocabulary)), dtype=int)
                tmp = []
                for word in document:
                    if x[myIndexVocabulary[word]] == 0:
                        x[myIndexVocabulary[word]] += 1
                        tmp.append(myIndexVocabulary[word])
                a = np.array(x).dot(weight) + b
                if y*a <= 0:
                    for i in tmp:
                        weight[i] += y*x[i]
                        u[i] += y * c * x[i]
                    b += y
                    B += y * c
                c += 1
    
    return weight - (1.0/c)*u , b - (1.0/c)*B

def main(model_file,avg_model_file):

    documentsPosNeg = {}
    negative = []
    positive = []

    documentsTruDec = {}
    deceptive = []
    truthful = []

    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    trainingPaths = defaultdict(list)
    for f in all_files:
        class1, class2, fold, fname = f.split('/')[-4:]
        trainingPaths[class1 + class2].append(f)

    negDecDoc, negDecWords, negTruDoc, negTruWords, posDecDoc, posDecWords, posTruDoc, posTruWords = processFold(trainingPaths)

    allText = []

    allText += negDecWords + negTruWords + posDecWords + posTruWords

    positive += posDecDoc + posTruDoc
    negative += negDecDoc + negTruDoc
    deceptive += negDecDoc + posDecDoc
    truthful += negTruDoc + posTruDoc

    documentsPosNeg['positive'] = positive
    documentsPosNeg['negative'] = negative
    documentsTruDec['truthful'] = truthful
    documentsTruDec['deceptive'] = deceptive

    vocabulary = getVocabulary(allText)
    myvocabulary = vocabulary.keys()
    myIndexVocabulary = {v: k for k, v in enumerate(myvocabulary)}

    weightVanillaPosNeg, biasVanillaPosNeg = trainingVanilla(documentsPosNeg, 100, myvocabulary, myIndexVocabulary,{'positive':1,'negative':-1})
    weightVanillaTruDec, biasVanillaTruDec = trainingVanilla(documentsTruDec, 100, myvocabulary, myIndexVocabulary,{'truthful':1,'deceptive':-1})
    weightAveragePosNeg, biasAveragePosNeg = trainingAverage(documentsPosNeg, 100, myvocabulary, myIndexVocabulary,{'positive':1,'negative':-1})
    weightAverageTruDec, biasAverageTruDec = trainingAverage(documentsTruDec, 100, myvocabulary, myIndexVocabulary,{'truthful':1,'deceptive':-1})

    with open(model_file, 'w+') as file:
        tmp = [list(weightVanillaPosNeg), biasVanillaPosNeg, list(weightVanillaTruDec), biasVanillaTruDec, myvocabulary, myIndexVocabulary]
        file.write(json.dumps(tmp))

    with open(avg_model_file,'w+') as file:
        tmp = [list(weightAveragePosNeg), biasAveragePosNeg, list(weightAverageTruDec), biasAverageTruDec, myvocabulary, myIndexVocabulary]
        file.write(json.dumps(tmp))


if __name__ == "__main__":
    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt"
    main(model_file,avg_model_file)
