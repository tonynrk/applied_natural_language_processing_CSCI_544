import json
from collections import Counter
import re
import codecs
import sys

def main(input_file,model_file):
    emitprob = {}
    transprob = {}
    initprob = {}

    # for using transprob
    tag = []

    with codecs.open(input_file, encoding='utf-8') as file:
        # read all file
        for line in file:
            # give every word to be lowercase

            tokens = re.split('[\s]+',line.strip())
            # tokens = line.lower().strip().split(" ")

            # get the initprob with no denom
            start = tokenizeforslash(tokens[0])
            if start[1] not in initprob:
                initprob[start[1]] = 1
            else:
                initprob[start[1]] += 1

            # get the emitprob with no denom and knows all the tag
            for i in range(len(tokens)):
                token = tokenizeforslash(tokens[i])

                lowerword = token[0].lower()
                if lowerword not in emitprob:
                    emitprob[lowerword] = {token[1]:1}
                else:
                    if token[1] not in emitprob[lowerword]:
                        emitprob[lowerword][token[1]] = 1
                    else:
                        emitprob[lowerword][token[1]] = emitprob[lowerword][token[1]] + 1
                tag.append(token[1])

            # get the transprob with no denom
            for i in range(len(tokens)-1):
                tokenGiven = tokenizeforslash(tokens[i])
                tokenProb = tokenizeforslash(tokens[i+1])
                if tokenGiven[1] not in transprob:
                    transprob[tokenGiven[1]] = {tokenProb[1]:1}
                else:
                    if tokenProb[1] not in transprob[tokenGiven[1]]:
                        transprob[tokenGiven[1]][tokenProb[1]] = 1
                    else:
                        transprob[tokenGiven[1]][tokenProb[1]] += 1

    tags = Counter(tag)
    uniqueTag = tags.keys()


    # get the initprob with denom, suppose we add 1 smoothing for all transition in any given
    initprobDenom = sum(initprob.values())

    for tag in uniqueTag:
        if tag not in initprob:
            initprob[tag] = 1
        else:
            initprob[tag] += 1
    for var in initprob:
        initprob[var] = float(initprob[var])/(initprobDenom + len(uniqueTag))

    # get the emitprob with denom

    for var in emitprob:
        for var2 in emitprob[var]:
            emitprob[var][var2] = float(emitprob[var][var2]) / tags[var2]



    # get the transprob with denom, suppose we add 1 smoothing for all transition in any given

    denomTransProb = {}
    for var in transprob:
        denomTransProb[var] = sum(transprob[var].values())
    for var in transprob:
        for tag in uniqueTag:
            if tag not in transprob[var]:
                transprob[var][tag] = 1
            else:
                transprob[var][tag] += 1
    for var in transprob:
        for var2 in transprob[var]:
            transprob[var][var2] = float(transprob[var][var2]) / (denomTransProb[var] + len(uniqueTag))


    # get the transprob with adding the left tag
    leftTransprob = list(set(uniqueTag) - set(transprob.keys()))

    if leftTransprob != []:
        for var in leftTransprob:
            tagprob = {}
            for tag in uniqueTag:
                tagprob[tag] = 0
            transprob[var] = tagprob

    with codecs.open(model_file, 'w+',encoding='utf-8') as file:
        tmp = [initprob,transprob,emitprob,uniqueTag,tags]
        file.write(json.dumps(tmp , indent = 2))


def tokenizeforslash(tmp):
    subtmp = 0
    for i in range(len(tmp))[::-1]:
        if tmp[i] == "/":
            subtmp = i
            break
    return [tmp[0:subtmp],tmp[subtmp+1:]]



if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = "hmmmodel.txt"
    main(input_file, model_file)
