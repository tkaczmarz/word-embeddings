from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding="utf-8"):
            yield line.split()


train_data = "data/news.en-00000-of-00100.txt"
print("Training model on (" + train_data + ")...")
model = Word2Vec(MySentences(train_data), workers=8, size=100)
print("Training finished. Saving...")
model.save('model')
print("Model saved!")
