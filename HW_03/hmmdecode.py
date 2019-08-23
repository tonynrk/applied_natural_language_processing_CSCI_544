import json
import numpy as np
import re
import codecs
import sys


def main(testing_file,model_file,output_file):
    [initprob,transprob,emitprob,uniqueTag,tags] = json.loads(open(str(model_file)).read())
    indexUniqueTag = {k: v for k, v in enumerate(uniqueTag)}
    POSSequence = []
    with codecs.open(testing_file, encoding='utf-8') as file:
        # read all file
        for line in file:
            tokens = re.split('[\s]+', line.lower().strip())
            tagOfLine = decoding(tokens,initprob,transprob,emitprob,uniqueTag,indexUniqueTag,tags)

            realTokens = re.split('[\s]+', line.strip())
            POSofSequence = zip(realTokens,tagOfLine)

            realPOSofSequence = ""

            for x,y in POSofSequence:
                realPOSofSequence += x+"/"+y+" "

            realPOSofSequence = realPOSofSequence.rstrip()

            POSSequence.append(realPOSofSequence)

    with codecs.open(output_file, 'w+',encoding='utf-8') as file:
        for var in POSSequence:
            file.write(var+'\n')

            # with open(output_file, 'w+') as file:
            #      file.write(realPOSofSequence+'\n')



def decoding(tokens,initprob,transprob,emitprob,uniqueTag,indexUniqueTag,tags):

    biggestTag = max(tags, key=tags.get)
    
    n = len(tokens)
    T = len(uniqueTag)
    viterbi = []
    for i in range(T):
        viterbi.append([0] * n)
    backpointer = []
    for i in range(T):
        backpointer.append([0]*n)

    for i in range(T):
        if tokens[0] in emitprob:
            if indexUniqueTag[i] in emitprob[tokens[0]]:
                viterbi[i][0] = initprob[indexUniqueTag[i]]*emitprob[tokens[0]][indexUniqueTag[i]]
            else:
                viterbi[i][0] = 0
        else:
            viterbi[i][0] = initprob[indexUniqueTag[i]]* float(tags[indexUniqueTag[i]]) / sum(tags.values())

        backpointer[i][0] = 0



    for i in range(1,n):
        for j in range(T):
            tmp = [0]*T
            for x in range(T):
                if tokens[i] in emitprob:
                    if indexUniqueTag[j] in emitprob[tokens[i]]:
                        tmp[x] = viterbi[x][i-1] * transprob[indexUniqueTag[x]][indexUniqueTag[j]] * emitprob[tokens[i]][indexUniqueTag[j]]
                    else:
                        tmp[x] = 0
                else:
                    tmp[x] = viterbi[x][i-1] * transprob[indexUniqueTag[x]][indexUniqueTag[j]] * float(tags[indexUniqueTag[j]]) / sum(tags.values())


            viterbi[j][i] = max(np.array(tmp))
            backpointer[j][i] = np.argmax(np.array(tmp))



    viterbi = np.array(viterbi)


    # bestpathprob = max(viterbi[:,-1])

    bestpathpointer = int(np.argmax(viterbi[:,-1]))

    bestpath = []
    bestpath.insert(0,bestpathpointer)

    for i in range(n)[::-1][:-1]:
        bestpath.insert(0,backpointer[bestpathpointer][i])
        bestpathpointer = backpointer[bestpathpointer][i]


    tagOfSpeech = []
    for var in bestpath:
        tagOfSpeech.append(indexUniqueTag[var])

    return tagOfSpeech

if __name__=="__main__":
    testing_file = sys.argv[1]
    model_file = "hmmmodel.txt"
    output_file = "hmmoutput.txt"
    main(testing_file,model_file,output_file)

