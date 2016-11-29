# coding=utf-8

import sys
import parser as ps
import counter as ct
import cPickle as pickle
from math import log
from gensim.models import word2vec

THRESHOLD = 0.08
VERSION = 'sklearn'

def load_file():
    # process data
    if sys.argv[1][-1] != '/':
        sys.argv[1] += '/'
    title = open(sys.argv[1] + 'title_StackOverflow.txt', 'r').read()
    check = open(sys.argv[1] + 'check_index.csv', 'r').read().split('\n')[1:-1]
    docs = open(sys.argv[1] + 'docs.txt', 'r').read()

    for i in range(len(check)):
        check[i] = [int(x) for x in check[i].split(',')[1:]]
    print check[0], check[-1]

    docs = ps.removeUselessContent(docs)
    print docs[:40], docs[-40:]

    return title, check, docs

def main(threshold=THRESHOLD) :
    ## load_data ##
    title, check, docs = load_file()


    ## parse data ##
    title_docs = title.split('\n')[:-1]
    # parse titles to documents
    for i in range(len(title_docs)):
        title_docs[i] = ps.removeUselessContent(title_docs[i])

    if VERSION == 'sklearn':
        for i in range(len(title_docs)):
            title_docs[i] = " ".join(title_docs[i])
    else:
        # count tf & idf of corpus
        tmpTitle = ps.removeUselessContent(title)
        # Terms, Model = ps.generalModel(docs+tmpTitle, title_docs)
        Terms, Model = pickle.load(open("model/terms_ver_cosine.pkl", "rb"))
        pickle.dump((Terms, Model), open("model/terms_ver_cosine.pkl", "wb"), True)

        # count tf of documents
        title_models = []
        for i in range(len(title_docs)):
            terms, model = ps.generalModel(title_docs[i])
            model = ps.parseTFIDF(terms, model, Model)
            title_models.append({'terms':terms, 'tfidf':model['tfidf']})
        # print title_models[0]['tf']['a'],title_models[0]['length']
        pickle.dump(title_models, open("model/title_models.pkl", "wb"), True)


    # test documents pairs
    out = open(sys.argv[2], 'w')
    out.write('ID,Ans\n')
    Min = float("inf")
    Max = -float("inf")
    MinPos = MaxPos = 0
    Yes = No = 0
    Sum = 0
    if VERSION == 'sklearn':
        print "generating cosine matrix......"
        cosineMatrix = ct.cosineMatrix(title_docs)
        for i in range(len(check)):
            doc1 = check[i][0]
            doc2 = check[i][1]
            cosineSimilarity = cosineMatrix[doc1][doc2]
            print i, cosineSimilarity
            if cosineSimilarity < Min:
                Min = cosineSimilarity
                MinPos = i
            if cosineSimilarity > Max:
                Max = cosineSimilarity
                MaxPos = i
            Sum += cosineSimilarity
            if cosineSimilarity >= threshold:
                Yes += 1
                out.write(str(i) + ',' + str(1) + '\n')
            else:
                No += 1
                out.write(str(i) + ',' + str(0) + '\n')
    else:
        for i in range(len(check)):
            doc1 = check[i][0]
            doc2 = check[i][1]
            cosineSimilarity = ct.docCosineSimilarity(title_models[doc1], title_models[doc2], title_docs[doc1], title_docs[doc2])
            print i, cosineSimilarity
            if cosineSimilarity < Min:
                Min = cosineSimilarity
                MinPos = i
            if cosineSimilarity > Max:
                Max = cosineSimilarity
                MaxPos = i
            Sum += cosineSimilarity
            if cosineSimilarity >= threshold:
                Yes += 1
                out.write(str(i) + ',' + str(1) + '\n')
            else:
                No += 1
                out.write(str(i) + ',' + str(0) + '\n')
    print("MinPos:%d, Min:%f" % (MinPos, Min))
    print("MaxPos:%d, Max:%f" % (MaxPos, Max))
    print("Yes:%d, No:%d" % (Yes, No))
    print("Sum:%f, Mean:%f" % (Sum, Sum/len(check)))

if __name__ == '__main__':
    main()
