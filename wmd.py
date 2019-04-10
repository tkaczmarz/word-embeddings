from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk.corpus import stopwords
from nltk import download
import logging
import time

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'
sentence_band = 'The band gave a concert in Japan'
obama_speaks = 'Obama speaks in Illinois'
sentence_obama_proc = sentence_obama.lower().split()
sentence_president_proc = sentence_president.lower().split()
sentence_band_proc = sentence_band.lower().split()
obama_speaks_proc = obama_speaks.lower().split()

download('stopwords')

# Remove stopwords.
stop_words = stopwords.words('english')
sentence_obama_proc = [w for w in sentence_obama_proc if w not in stop_words]
sentence_president_proc = [w for w in sentence_president_proc if w not in stop_words]
sentence_band_proc = [w for w in sentence_band_proc if w not in stop_words]
obama_speaks_proc = [w for w in obama_speaks_proc if w not in stop_words]


def dist(doc1, doc2):
    print(doc1 + ' -> ' + doc2)
    d = model.wmdistance(doc1, doc2)
    print(str(d))


print('loading model...')
# model = Word2Vec.load('models/billion-model')
polish = True

if not polish:
    file = 'models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    model.init_sims(replace=True)  # normalize vectors
    print('VOCAB LEN: ' + str(len(model.vocab)))
    # i = 0
    # for word in model.vocab:
    #     print('Word: ' + str(word))
    #     i += 1
    #     if i > 200:
    #         break
    print(sentence_obama)
    distance = model.wmdistance(sentence_president_proc, sentence_obama_proc)
    print(str(distance))
    print(sentence_president)
    distance = model.wmdistance(sentence_band_proc, sentence_president_proc)
    print(str(distance))
    print(sentence_band)

    print('\n' + sentence_president)
    distance = model.wmdistance(obama_speaks_proc, sentence_president_proc)
    print(str(distance))
    print(obama_speaks)
else:
    file = 'models/polish-converted100.bin'
    model = KeyedVectors.load_word2vec_format(file, binary=True, encoding='utf8')
    model.init_sims(replace=True)  # normalize vectors
    print('VOCAB LEN: ' + str(len(model.vocab)))
    # i = 0
    # for word in model.vocab:
    #     print('Word: ' + str(word))
    #     i += 1
    #     if i > 20:
    #         break
    dist('chleb', 'bułka')
    dist('chleb', 'samochód')
    dist('Prezydent przemawiał przed publicznością w Radomiu.', 'Prezydent udzielił wywiadu reporterom w Gdańsku.')
    dist('Prezydent przemawiał przed publicznością w Radomiu.', 'Zespół dał koncert w Japonii.')
