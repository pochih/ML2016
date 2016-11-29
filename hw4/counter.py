from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy import spatial
from math import log, sqrt
import numpy as np

SMOOTHING = 0.00027
MIN_DF = 3

def wordProbability(word, model, Terms):
    if word in model['tf']:
        wordInModel = model['tf'][word]
    else:
        wordInModel = 0
    modelLength = model['length']
    # print word, float(SMOOTHING+wordInModel), "/", float(SMOOTHING*len(Terms)+modelLength), float(SMOOTHING+wordInModel) / float(SMOOTHING*len(Terms)+modelLength)
    return float(SMOOTHING+wordInModel) / float(SMOOTHING*len(Terms)+modelLength)

def countProbability(doc, model, Terms, smooth=None):
    if smooth != None:
        SMOOTHING = smooth
    probability_pi = 0
    for word in doc:
        probability_pi += log(wordProbability(word, model, Terms))
    return probability_pi

def cosineMatrix(content, min_df=MIN_DF):
    vect = TfidfVectorizer(min_df)
    tfidf = vect.fit_transform(content)

    return (tfidf * tfidf.T).A

def docCosineSimilarity(model1, model2, doc1, doc2):
    query = model2['terms']
    document = model1['terms']
    dot = 0
    for term in query:
        if term in document:
            dot += (model1['tfidf'][term] * model2['tfidf'][term])
    query_abs = 0
    document_abs = 0
    for term in doc1:
        document_abs += (model1['tfidf'][term] * model1['tfidf'][term])
    for term in doc2:
        query_abs += (model2['tfidf'][term] * model2['tfidf'][term])
    return dot / (sqrt(query_abs) * sqrt(document_abs))

def vecCosineSimilarity(vec1, vec2, ver='scipy'):
    if ver == 'scipy':
        return 1 - spatial.distance.cosine(vec1, vec2)
    elif ver == 'sklearn':
        A = np.array([vec1, vec2])
        A_sparse = sparse.csr_matrix(A)
        return cosine_similarity(A_sparse)[0][1]
    else:
        dot = vec1_abs = vec2_abs = 0
        for i in range(len(vec1)):
            dot += (vec1[i] * vec2[i])
            vec1_abs += (vec1[i] * vec1[i])
            vec2_abs += (vec2[i] * vec2[i])
        return dot / (sqrt(vec1_abs) * sqrt(vec2_abs))
