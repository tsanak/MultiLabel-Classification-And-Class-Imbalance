import json
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import os.path

nltk.download('stopwords')
stemmer = SnowballStemmer("english")

bow = CountVectorizer(ngram_range=(1, 2))
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)

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
    if not os.path.isfile('./ml_preprocessed.csv'):
        data = pd.read_csv('../dataset/full_movies_metadata.csv', sep=',')

        data['overview'] = data['title'] + ' ' + data['overview']

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

        data = pd.DataFrame(data)
        data.to_csv('ml_preprocessed.csv', index=False, sep=",")
    else:
        data = pd.read_csv("./ml_preprocessed.csv")
    return data


def sklearn_train_test(pd_df):
    df = pd_df
    X = df['overview']
    y = df[df.columns[1:]]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, test_size=0.2)

    return x_train, x_test, y_train, y_test

