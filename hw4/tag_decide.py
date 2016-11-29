# coding=utf-8

import sys
import parser as ps
import counter as ct
import cPickle as pickle
from math import log

THRESHOLD = -30
SMOOTHING = 0.00027

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
    # load_data
    title, check, docs = load_file()
    title_docs = title.split('\n')[:-1]
    ## parse titles to documents
    for i in range(len(title_docs)):
        title_docs[i] = ps.removeUselessContent(title_docs[i])
    tags = open("tags.txt", "r").read().split("\n")

    # count tf & idf of corpus
    tmpTitle = ps.removeUselessContent(title)
    # Terms, Model = ps.generalModel(docs+tmpTitle, title_docs)
    Terms, Model = pickle.load(open("model/terms_ver_NB.pkl", "rb"))

    # count tf of documents
    title_models = []
    for i in range(len(title_docs)):
        term, model = ps.generalModel(title_docs[i])
        title_models.append({'term':term, 'tf':model['tf'], 'length':len(title_docs[i])})
        title_models[i]['tags'] = []
        for j in range(len(tags)):
            if tags[j] in title_docs[i]:
                title_models[i]['tags'].append(j)
    print title_models[0]['tf']['a'],title_models[0]['length'],title_models[0]['tags']

    pickle.dump((Terms, Model), open("model/terms_ver_NB.pkl", "wb"), True)
    pickle.dump(title_models, open("model/title_models.pkl", "wb"), True)

    # test documents pairs
    out = open(sys.argv[2], 'w')
    out.write('ID,Ans\n')
    Min = float("inf")
    Max = -float("inf")
    MinPos = MaxPos = 0
    Yes = No = 0
    Sum = 0
    for i in range(len(check)):
        doc1 = check[i][0]
        doc2 = check[i][1]
        prob = ct.countProbability(title_docs[doc1], title_models[doc2], Terms, smooth=SMOOTHING)
        print i, prob
        if prob < Min:
            Min = prob
            MinPos = i
        if prob > Max:
            Max = prob
            MaxPos = i
        Sum += prob
        prob = -float("inf")
        for tag in title_models[doc1]['tags']:
            if tag in title_models[doc2]['tags']:
                print "Tag found!"
                prob = 1
        if prob >= threshold:
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
