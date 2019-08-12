import sys
import glob
import os
import re
import json
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



def testing(document,weight,b,myvocabulary,myIndexVocabulary):
    with open(document) as file:
        w = file.read().rstrip()
    words = tokenization(w)
    x = np.zeros((len(myvocabulary)), dtype=int)
    for word in words:
        if word in myIndexVocabulary:
            if x[myIndexVocabulary[word]] == 0:
                x[myIndexVocabulary[word]] += 1
    a = np.array(x).dot(weight) + b
    if a == 0.0:
        a = 1
    return int(np.sign(a))



def testingoutput(testingPaths, weightPosNeg, biasPosNeg, weightTruDec, biasTruDec, myvocabulary, myIndexVocabulary):

    tmp = []
    for document in testingPaths:
        classTruDec = testing(document,weightTruDec,biasTruDec,myvocabulary,myIndexVocabulary)
        classPosNeg = testing(document,weightPosNeg,biasPosNeg,myvocabulary,myIndexVocabulary)
        if classTruDec == 1:
            classTruDec = 'truthful'
        else:
            classTruDec = 'deceptive'
        if classPosNeg == 1:
            classPosNeg = 'positive'
        else:
            classPosNeg = 'negative'

        tmp.append(classTruDec + " " + classPosNeg + " " + document + "\n")

    return tmp


def main(model_file,input_path,output_file):

    [weightPosNeg, biasPosNeg, weightTruDec, biasTruDec, myvocabulary, myIndexVocabulary] = json.loads(open(str(model_file)).read())
    
    weightPosNeg = np.array(weightPosNeg)
    weightTruDec = np.array(weightTruDec)

    all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))

    result = testingoutput(all_files, weightPosNeg, biasPosNeg, weightTruDec, biasTruDec, myvocabulary, myIndexVocabulary)

    with open(output_file, 'w+') as file:
        for sentence in result:
            file.write(sentence)

    return

if __name__ == "__main__":
    model_file = sys.argv[1]
    output_file = "percepoutput.txt"
    input_path = sys.argv[2]
    main(model_file,input_path,output_file)