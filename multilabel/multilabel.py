import json
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

nltk.download('stopwords')
stemmer = SnowballStemmer("english")

bow = CountVectorizer(ngram_range=(1, 2))
tfidf = TfidfVectorizer(ngram_range=(1, 2))

stop_words = set(stopwords.words('english'))
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)


def remove_stop_words(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


def stemming(sentence):
    stem_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stem_sentence += stem
        stem_sentence += " "
    stem_sentence = stem_sentence.strip()
    return stem_sentence


def clean_punctuation(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def bag_of_words(x_train, x_test):
    x_train = bow.fit_transform(x_train)
    x_test = bow.transform(x_test)
    return x_train, x_test


def tf_idf(x_train, x_test):
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)
    return x_train, x_test


def preprocess():
    mlb = MultiLabelBinarizer()
    # initialize Binary Relevance multi-label classifier
    # with an SVM classifier
    # SVM in scikit only supports the X matrix in sparse representation

    data = pd.read_csv('../dataset/full_movies_metadata.csv', sep=',')

    # Remove rows with no labels and find unique labels
    rowsWithNoLabel = []
    genreList = []
    dontAddGenres = ['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions', 'Aniplex',
                     'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks',
                     'Odyssey Media',
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

    for row in data['genres']:
        fix_json = row.replace("\'", "\"")
        genre = json.loads(fix_json)
        current_genre_list = [g['name'] for g in genre]
        for g in genreList:
            if g in current_genre_list:
                genreDict[g].append(1)
            else:
                genreDict[g].append(0)

    data['overview'] = data['overview'].str.lower()
    data['overview'] = data['overview'].apply(clean_punctuation)
    data['overview'] = data['overview'].apply(remove_stop_words)
    data['overview'] = data['overview'].apply(stemming)

    data = {
        'overview': data['overview']
    }

    for g in genreDict:
        data[g] = np.array(genreDict[g])

    # Create DataFrame
    data = pd.DataFrame(data)
    data = data.sample(n=5000, random_state=0)
    return data


def sklearn_train_test(pd_df):
    df = pd_df
    X = df['overview']
    y = df[df.columns[1:]]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, test_size=0.2)

    return x_train, x_test, y_train, y_test

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

# SVC n=5.000 BoW
# Precision: 0.677895
# Recall: 0.152174
# F1: 0.248553

# SVC n=5000 TF-idf
# Precision: 0.677966
# Recall: 0.132325
# F1: 0.221431
