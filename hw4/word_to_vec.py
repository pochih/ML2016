# coding=utf-8

import sys
import numpy as np
import parser as ps
import counter as ct
import math
import gensim
# from nltk.corpus import stopwords
from math import log
from gensim.models import word2vec

SIMILARITY = 'Cosine'
THRESHOLD = 0.864
# SIMILARITY = 'WMD'
# THRESHOLD = 3.95
VEC_SIZE = 200

def load_file():
    # process data
    if sys.argv[1][-1] != '/':
        sys.argv[1] += '/'
    title = open(sys.argv[1] + 'title_StackOverflow.txt', 'r').read()
    check = open(sys.argv[1] + 'check_index.csv', 'r').read().split('\n')[1:-1]
    docs = open(sys.argv[1] + 'docs.txt', 'r').read()

    for i in range(len(check)):
        check[i] = [int(x) for x in check[i].split(',')[1:]]
    # print check[0], check[-1]

    docs = ps.removeUselessContent(docs)
    # print docs[:40], docs[-40:]

    return title, check, docs

def load_stopwords():
    sw = open('./stopword.txt', 'r').read().split('\n')
    return sw

def checkAnswer(answer):
    if math.isnan(answer):
        return False
    if SIMILARITY == 'Cosine':
        if answer >= THRESHOLD:
            return True
        else:
            return False
    elif SIMILARITY == 'WMD':
        if answer <= THRESHOLD:
            return True
        else:
            return False

def main(threshold=THRESHOLD) :
    ## load_data ##
    title, check, docs = load_file()
    stopwords = load_stopwords()


    ## parse data ##
    # parse titles to documents
    title_docs = title.split('\n')[:-1]
    for i in range(len(title_docs)):
        title_docs[i] = ps.removeUselessContent(title_docs[i])
        title_docs[i] = [w for w in title_docs[i] if w not in stopwords]
    tmpTitle = ps.removeUselessContent(title)
    tmpout = open('model/words.txt', 'w')
    count = 0
    for word in docs+tmpTitle:
        if word in stopwords:
            continue
        tmpout.write(word + ' ')
        count += 1
        if count % 100 == 0:
            tmpout.write('\n')


    ## training ##
    # build word vector
    corpus = word2vec.Text8Corpus("model/words.txt")
    model = word2vec.Word2Vec(corpus, size=VEC_SIZE)
    model.save_word2vec_format(u"model/title_vector.txt", binary=False)
    model.save_word2vec_format(u"model/title_vector.bin", binary=True)


    ## load model ##
    # wordVec = word2vec.Word2Vec.load_word2vec_format('model/title_vector.txt', binary=False)
    wordVec = word2vec.Word2Vec.load_word2vec_format('model/title_vector.bin', binary=True)
    # vector_file = open('model/title_vector.txt', 'r').read().split("\n")[1:-1]
    # vector_file = [x.split() for x in vector_file]
    # wordVec = {}
    # for i in range(len(vector_file)):
    #     vector_file[i][1:] = [float(x) for x in vector_file[i][1:]]
    #     wordVec[vector_file[i][0]] = np.array(vector_file[i][1:])
    # # print wordVec[vector_file[0][0]], wordVec[vector_file[-1][0]]
    titleVec = []
    for i in range(len(title_docs)):
        tmpVec = np.zeros((VEC_SIZE))
        for term in title_docs[i]:
            if term not in wordVec or term in stopwords:
                continue
            tmpVec += wordVec[term]
        titleVec.append(tmpVec)
    # print 'titleVec[0]', titleVec[0], 'titleVec[-1]', titleVec[-1]


    ## test documents pairs ##
    out = open(sys.argv[2], 'w')
    score = open('score', 'w')
    out.write('ID,Ans\n')
    Min = float("inf")
    Max = -float("inf")
    MinPos = MaxPos = 0
    Yes = No = 0
    Sum = 0.
    if SIMILARITY == 'Cosine':
        print('=== Using Cosine Similarity, threshold: %f ===' % (threshold))
    elif SIMILARITY == 'WMD':
        print("=== Using Word Mover's Distance, threshold: %f ===" % (threshold))
    for i in range(len(check)):
        doc1 = check[i][0]
        doc2 = check[i][1]
        if SIMILARITY == 'Cosine':
            # cosine similarity
            answer = ct.vecCosineSimilarity(titleVec[doc1], titleVec[doc2])
        elif SIMILARITY == 'WMD':
            # Word Mover's Distance
            answer = wordVec.wmdistance(title_docs[doc1], title_docs[doc2])
        if math.isnan(answer) == False and answer != float("inf"):
            Sum += answer
        if math.isnan(answer) == False:
            score.write(str(answer) + '\n')
        else:
            score.write(str(float("inf")) + '\n')
        if i % 50000 == 0:
            print 'producing index', i, 'answer:', answer
        if answer < Min:
            Min = answer
            MinPos = i
        if answer > Max:
            Max = answer
            MaxPos = i
        if checkAnswer(answer) == True:
            Yes += 1
            out.write(str(i) + ',' + str(1) + '\n')
        else:
            No += 1
            out.write(str(i) + ',' + str(0) + '\n')
    out.close()
    print("MinPos:%d, Min:%f" % (MinPos, Min))
    print("MaxPos:%d, Max:%f" % (MaxPos, Max))
    print("Yes:%d, No:%d" % (Yes, No))
    print("Sum:%f, Mean:%f" % (Sum, Sum/len(check)))

if __name__ == '__main__':
    main()
