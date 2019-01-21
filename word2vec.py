from gensim.models import Word2Vec
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec.load('billion-model')

print('sim(Berlin, Germany) = ' + str(model.similarity('Berlin', 'Germany')))
print('sim(strong, France) = ' + str(model.similarity('strong', 'France')))

print('most_similar(positive=[woman, king], negative=[man]) = ' +
      str(model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)))

print("doesnt_match(Spain Greece Cairo Egypt) = " + str(model.doesnt_match("Spain Greece Cairo Egypt".split())))
print("doesnt_match(France Paris Berlin Cairo) = " + str(model.doesnt_match("France Paris Berlin Cairo".split())))
