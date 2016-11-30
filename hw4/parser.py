import re
import time
from math import log

def removeUselessContent(content):

    ret = []  # return list
    content = content.split()

    for i in range(0, len(content)):
        # vocab = re.findall('[a-zA-Z0-9]+', content[i])
        vocab = re.findall('[a-zA-Z]+', content[i])
        content[i] = ''.join(vocab)
        content[i] = content[i].lower()
        if content[i] != '':
            ret.append(content[i])

    return ret

def generalModel(content, documents=None):

    Terms = []
    Model = {'tf':{}}
    # print("parse TF & terms...")
    ts = time.time()
    for i in range(len(content)):
        term = content[i]
        if term not in Terms:
            Terms.append(term)
        if term not in Model['tf']:
            Model['tf'][term] = 1
        else:
            Model['tf'][term] += 1
    # normalize tf
    for key in Model['tf']:
        Model['tf'][key] /= float(len(content))
    te = time.time()
    # print(te-ts, "secs")

    if documents != None:
        # print("parse IDF...")
        ts = time.time()
        Model['idf'] = {}
        for i in range(len(Terms)):
            doc_count = 0.
            for j in range(len(documents)):
                if Terms[i] in documents[j]:
                    doc_count += 1
            # normalize idf
            if doc_count != 0:
                Model['idf'][Terms[i]] = 1 + log(len(documents) / doc_count)
            else:
                Model['idf'][Terms[i]] = 1
        te = time.time()
        # print(te-ts, "secs")

    return Terms, Model

def parseTFIDF(terms, model, Model):
    model['tfidf'] = {}
    for term in terms:
        if term not in model['tfidf']:
            tf = model['tf'][term]
            idf = Model['idf'][term]
            model['tfidf'][term] = tf * idf
    return model

def removeStopwords(word_list, stopwords):
    ret = []
    for word in word_list:
        if word not in stopwords:
            ret.append(word)
    return ret