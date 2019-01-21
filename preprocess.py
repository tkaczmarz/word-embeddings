# Pre-processing a document.
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords

download('stopwords')
download('punkt')  # Download data for tokenizer.
stop_words = stopwords.words('english')


def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


w2v_corpus = []
input_data = 'data/1-billion-words.txt'
output_data = 'data/1-billion-words-preprocessed.txt'
counter = 0
# writing preprocessed data to file
with open(input_data, encoding="utf8") as input_file:
    with open(output_data, 'w', encoding="utf8") as output_file:
        for line in input_file:
            counter += 1
            # w2v_corpus.append(line)
            output_file.write(str(preprocess(line)))
            if counter % 10000 == 0:
                print("Progress: " + str(counter) + " lines")
                break

# reading preprocessed data
# with open(output_data, 'r', encoding="utf8") as output_file:
#     w2v_corpus = output_file.read().splitlines()
