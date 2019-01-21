# Pre-processing a document.
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import json
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity


def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


download('stopwords')
download('punkt')  # Download data for tokenizer.
stop_words = stopwords.words('english')

# Business IDs of the restaurants.
ids = ['iCQpiavjjPzJ5_3gPD5Ebg', 'pomGBqfbxcqPv14c3XH-ZQ', 'jtQARsP6P-LbkyjbO1qNGg',
       'elqbBhBfElMNSrjFqW3now', 'Ums3gaP2qM3W1XcA5r6SsQ', 'vgfcTvK81oD4r50NMjU2Ag']

w2v_corpus = []  # Documents to train word2vec on (all 6 restaurants).
wmd_corpus = []  # Documents to run queries against (only one restaurant).
documents = []  # wmd_corpus, with no pre-processing (so we can see the original documents).
counter = 0
with open('data/yelp_academic_dataset_review.json', encoding="utf8") as data_file:
    for line in data_file:
        json_line = json.loads(line)

        counter += 1
        if counter % 100000 == 0:
            print("Progress: " + str(counter))

        if json_line['business_id'] not in ids:
            # Not one of the 6 restaurants.
            continue

        # Pre-process document.
        text = json_line['text']  # Extract text from JSON object.
        text = preprocess(text)

        # Add to corpus for training Word2Vec.
        w2v_corpus.append(text)

        if json_line['business_id'] == ids[0]:
            # Add to corpus for similarity queries.
            wmd_corpus.append(text)
            documents.append(json_line['text'])


# Train Word2Vec on all the restaurants.
model = Word2Vec(w2v_corpus, workers=3, size=100)

# Initialize WmdSimilarity.
num_best = 10
instance = WmdSimilarity(wmd_corpus, model, num_best=10)

sent = 'Very good, you should seat outdoor.'
query = preprocess(sent)

sims = instance[query]  # A query is simply a "look-up" in the similarity class.

# Print the query and the retrieved documents, together with their similarities.
print('Query:')
print(sent)
for i in range(num_best):
    print()
    print('sim = %.4f' % sims[i][1])
    print(documents[sims[i][0]])
