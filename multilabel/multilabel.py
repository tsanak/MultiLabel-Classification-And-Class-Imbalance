import json
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

nltk.download('stopwords')
stemmer = SnowballStemmer("english")


def stemming(sentence):
    stem_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stem_sentence += stem
        stem_sentence += " "
    stem_sentence = stem_sentence.strip()
    return stem_sentence


def clean_punctuation(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def remove_stop_words(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


mlb = MultiLabelBinarizer()
# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

data = pd.read_csv('../dataset/full_movies_metadata.csv', sep=',')

# Remove rows with no labels and find unique labels
rowsWithNoLabel = []
genreList = []
dontAddGenres = ['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions', 'Aniplex',
                 'GoHands', 'BROSTA TV',  'Mardock Scramble Production Committee', 'Sentai Filmworks', 'Odyssey Media',
                 'Pulser Productions', 'Rogue State', 'The Cartel']
for index, g in enumerate(data['genres']):
    fix_json = g.replace("\'", "\"")
    genre = json.loads(fix_json)
    if len(genre) == 0:
        rowsWithNoLabel.append(index)
    for current_genre in genre:
        if current_genre['name'] not in genreList and current_genre['name'] not in dontAddGenres:
            genreList.append(current_genre['name'])

data.drop(data.index[rowsWithNoLabel], inplace=True)
# Remove rows that don't have overview
data.dropna(subset=['overview'], inplace=True)

# Create dictionary for each genre / label
genreDict = {}
for g in genreList:
    genreDict[g] = []
newTargets = []
for row in data['genres']:
    tmp_labels = [0 for i in genreList]
    fix_json = row.replace("\'", "\"")
    genre = json.loads(fix_json)
    current_genre_list = [g['name'] for g in genre]
    for g in genreList:
        if g in current_genre_list:
            genreDict[g].append(1)
        else:
            genreDict[g].append(0)
print(data['overview'])

data['overview'] = data['overview'].str.lower()
stop_words = set(stopwords.words('english'))
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

data['overview'] = data['overview'].apply(clean_punctuation)
data['overview'] = data['overview'].apply(remove_stop_words)
data['overview'] = data['overview'].apply(stemming)


print(data['overview'])

data = {
    'overview': data['overview']
}

for g in genreDict:
    data[g] = np.array(genreDict[g])

# Create DataFrame
data = pd.DataFrame(data)
data = data.sample(n=2500, random_state=0)
X = data['overview']
y = data[data.columns[1:]]

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, test_size=0.2)

vectorizer = CountVectorizer(ngram_range=(1, 2))

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
classifier = BinaryRelevance(
    classifier=MultinomialNB(),
    require_dense=[False, True]
)
# train
classifier.fit(x_train, y_train)

# predict
y_predicted = classifier.predict(x_test)

f1 = metrics.f1_score(y_test, y_predicted, average="micro")
# accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted, average="micro")
precision = metrics.precision_score(y_test, y_predicted, average="micro")

# print('Accuracy: %2f' % accuracy)
print('Precision: %2f' % precision)
print('Recall: %2f' % recall)
print('F1: %2f' % f1)

# Accuracy: 0.080000
# Precision: 0.026020
# Recall: 0.028022
# F1: 0.026984

# Multinomial NB n=10.000
# Precision: 0.693215
# Recall: 0.165376
# F1: 0.267045

# SVC
# Precision: 0.503546
# Recall: 0.163218
# F1: 0.246528
