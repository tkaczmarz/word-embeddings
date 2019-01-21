from gensim.models import KeyedVectors, Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding="utf-8"):
            yield line.split()


file = 'data/1-billion-words.txt'
sentences = MySentences(file)

model = Word2Vec(sentences, min_count=10, workers=8)
model.save('billion-model')

# model = KeyedVectors.load_word2vec_format('1-billion-words.tar.gz', binary=True)
# model.save('billion-model')


