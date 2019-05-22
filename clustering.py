from scipy.cluster.vq import whiten
from gensim.models import KeyedVectors
from gensim.corpora.dictionary import Dictionary
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import math
from numpy.random import permutation
import time
import wmd
import os

dirs = ['business', 'entertainment', 'politics', 'sport', 'tech']
fileCount = 0
docs = []
for folder in dirs:
    docs.append([])
    path = 'data/bbc/' + folder + '/'
    for filename in os.listdir(path):
        docs[len(docs) - 1].append(open(path + filename, "r").read())
        fileCount += 1
        # break
        if fileCount % 50 == 0:
            print('Processed ' + str(fileCount) + ' files')
            # break
print('Processed ' + str(fileCount) + ' files')
print('Preprocessing...')
# preprocess data
allWords = list()
for i in range(0, len(docs)):
    for j in range(0, len(docs[i])):
        docs[i][j] = wmd.preproc(docs[i][j])
        allWords += docs[i][j]
dict = Dictionary(documents=[allWords])

# create nbow vectors
print('Computing vectors...')
vectors = list()
for i in range(0, len(docs)):
    for j in range(0, len(docs[i])):
        vector = whiten(KeyedVectors.nbow(docs[i][j], dict))    # whitening increases accuracy
        # vector = KeyedVectors.nbow(docs[i][j], dict)
        vector = np.append(vector, i)   # add class column
        vectors.append(vector)

df = pd.DataFrame(vectors)
# print('DATA FRAME:\n' + str(df))

# split data to train and test sets
random_indices = permutation(df.index)
test_cutoff = math.floor(len(df) / 5)   # 80:20 ratio
test_set = df.loc[random_indices[1:test_cutoff]]
train_set = df.loc[random_indices[test_cutoff:]]

# choose x and y sets (x - vectors columns; y - class)
x_columns = list(range(0, len(vector) - 1))
y_column = len(vector) - 1

# CLUSTER (1-19 neighbors)
for n in range(1, 20):
    print(str(n) + ' neighbor(s):')
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_set[x_columns], train_set[y_column])
    predictions = knn.predict(test_set[x_columns])

    actual = test_set[y_column]
    rowCount = len(test_set)
    correctCount = 0
    for i in range(0, rowCount):
        if predictions[i] == actual._ndarray_values[i]:
            correctCount += 1

    print('Guessed ' + str(correctCount) + ' out of ' + str(rowCount) + ' (' + str(correctCount / rowCount) + '% accuracy)')
    mse = (((predictions - actual) ** 2).sum()) / len(predictions)
    print('Mean squared error: ' + str(mse))

#####################################################

# get data
# print('reading data')
# wmd.polish = False

# if wmd.polish:
#     doc1 = open('data/wikipl1.txt', "r", encoding="utf8").read()
#     doc2 = open('data/wikipl2.txt', "r", encoding="utf8").read()
#     doc3 = open('data/wikipl3.txt', "r", encoding="utf8").read()
#     sentence_obama = 'Prezydent przemawiał przed publicznością w Radomiu.'
#     sentence_president = 'Prezydent udzielił wywiadu reporterom w Gdańsku.'
# else:
#     # doc1 = open('data/tech1.txt', "r", encoding="utf8").read()
#     # doc2 = open('data/tech2.txt', "r", encoding="utf8").read()
#     # doc3 = open('data/sport1.txt', "r", encoding="utf8").read()
#     doc1 = open('data/wikien1.txt', "r", encoding="utf8").read()
#     doc2 = open('data/wikien2.txt', "r", encoding="utf8").read()
#     doc3 = open('data/wikien3.txt', "r", encoding="utf8").read()
#     sentence_obama = 'Obama speaks to the media in Illinois'
#     sentence_president = 'The president greets the press in Chicago'

# model = wmd.get_model()
#
# start = time.time()
# wmd.dist(model, doc1, doc2)
# print('Distance calculated in ' + str(time.time() - start) + ' seconds\n')
#
# start = time.time()
# wmd.dist(model, doc1, doc3)
# print('Distance calculated in ' + str(time.time() - start) + ' seconds\n')
#
# start = time.time()
# wmd.dist(model, doc1, sentence_obama)
# print('Distance calculated in ' + str(time.time() - start) + ' seconds\n')
#
# start = time.time()
# wmd.dist(model, sentence_president, sentence_obama)
# print('Distance calculated in ' + str(time.time() - start) + ' seconds\n')
#
# start = time.time()
# wmd.dist(model, sentence_president, sentence_president)
# print('Distance calculated in ' + str(time.time() - start) + ' seconds\n')
