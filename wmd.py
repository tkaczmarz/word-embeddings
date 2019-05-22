from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import word_tokenize
import logging
import time

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

polish = False

def dist(vectors, doc1, doc2, preprocess=True):
    # print(doc1 + ' -> ' + doc2)
    if preprocess:
        print('preprocessing first document...')
        doc1 = preproc(doc1)
        print('preprocessing second document...')
        doc2 = preproc(doc2)

    print('calculating distance...')
    d = vectors.wmdistance(doc1, doc2)
    print(' -----> Distance: ' + str(d))


def preproc(doc):
    doc_proc = word_tokenize(doc.lower())
    words = get_stopwords()#.split()
    doc_proc = [w for w in doc_proc if w not in words]
    doc_proc = [w for w in doc_proc if w.isalpha()]
    return doc_proc


def get_stopwords():
    if not polish:
        return open('data/stopwords/english.stopwords.txt', encoding='utf8').read()
    else:
        return open('data/stopwords/polish.stopwords.txt', encoding='utf8').read()


def get_model(normalize=True):
    if polish:
        print('loading Polish model...')
        model = KeyedVectors.load_word2vec_format('models/polish-converted100.bin', binary=True)
    else:
        print('loading English model...')
        model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    print('model loaded!')
    if normalize:
        model.init_sims(replace=True)
    return model


# print('loading model...')
# # model = Word2Vec.load('models/billion-model')
#
# if not polish:
#     sentence_obama = 'Obama speaks to the media in Illinois'
#     sentence_president = 'The president greets the press in Chicago'
#     sentence_band = 'The band gave a concert in Japan'
#     obama_speaks = 'Obama speaks in Illinois'
#
#     # print('Before: ' + sentence_obama)
#     # print('After: ' + str(preproc(sentence_obama)))
#
#     file = 'models/GoogleNews-vectors-negative300.bin'
#     model = KeyedVectors.load_word2vec_format(file, binary=True)
#     model.init_sims(replace=True)  # normalize vectors
#     # print('VOCAB LEN: ' + str(len(model.vocab)))
#     # i = 0
#     # for word in model.vocab:
#     #     print('Word: ' + str(word))
#     #     i += 1
#     #     if i > 20:
#     #         break
#
#     dist(model, sentence_president, sentence_obama)
#     dist(model, sentence_band, sentence_president)
#     dist(model, obama_speaks, sentence_president)
# else:
#     # print('Before: ' + 'Prezydent przemawiał przed publicznością w Radomiu')
#     # print('After: ' + str(preproc('Prezydent przemawiał przed publicznością w Radomiu')))
#
#     file = 'models/polish-converted100.bin'
#     model = KeyedVectors.load_word2vec_format(file, binary=True, encoding='utf8')
#     model.init_sims(replace=True)  # normalize vectors
#     # print('VOCAB LEN: ' + str(len(model.vocab)))
#     # i = 0
#     # for word in model.vocab:
#     #     print('Word: ' + str(word))
#     #     i += 1
#     #     if i > 20:
#     #         break
#
#     dist(model, 'chleb', 'bułka')
#     dist(model, 'chleb', 'samochód')
#     dist(model, 'Prezydent przemawiał przed publicznością w Radomiu', 'Prezydent udzielił wywiadu reporterom w Gdańsku')
#     dist(model, 'Prezydent przemawiał przed publicznością w Radomiu', 'Zespół dał koncert w Japonii')
