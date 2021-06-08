import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from imblearn.under_sampling import NearMiss
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek


nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

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


def preprocess():
    data = pd.read_csv('../dataset/reviews.csv', sep=',')

    data.dropna(subset=['Text'], inplace=True)
    data.dropna(subset=['Score'], inplace=True)

    data['Text'] = data['Text'].str.lower()
    data['Text'] = data['Text'].apply(clean_punctuation)
    data['Text'] = data['Text'].apply(remove_stop_words)
    # data['overview'] = data['overview'].apply(stemming)

    return data


def sklearn_train_test(pd_df, sampling=None):
    df = pd_df

    if sampling == 'undersample':
        dfs = []
        for i in range(1, 6):
            curr_df = df[df['Score'] == i]
            dfs.append(resample(curr_df, replace=False, n_samples=7500, random_state=0))

        df = pd.concat(dfs)


    X = df['Text']
    y = np.array([int(v) - 1 for v in df['Score'].values])
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, stratify=y, test_size=0.05)
    return x_train, x_test, y_train, y_test

def tf_idf(x_train, x_test):
    # convert th documents into a matrix
    X_train = tfidf.fit_transform(x_train)
    X_test = tfidf.transform(x_test)
    return X_train, X_test

def Smote(X_train, y_train):
    smt = SMOTE(random_state=0)
    X_smote, y_smote = smt.fit_resample(X_train, y_train)
    return X_smote, y_smote

def Near_miss(X_train, y_train):
    undersample = NearMiss(version=1, n_neighbors=3)
    X_near, y_near = undersample.fit_resample(X_train, y_train)
    return X_near, y_near

def tomek_links(X_train, y_train):
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_train, y_train)
    return X_res, y_res


def Smotetomek(X_train, y_train):
    smt = SMOTETomek(random_state=0)
    X_smtm, y_smtm = smt.fit_resample(X_train, y_train)
    return X_smtm, y_smtm

def confusion_matrix(y_actual, y_predicted):
    """ This method finds the number of True Positives, False Positives,
    True Negatives and False Negative between the hidden movies
    and those predicted by the recommendation algorithm
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_predicted)):
        if y_actual[i] == y_predicted[i] == 1:
            TP += 1
        if y_predicted[i] == 1 and y_actual[i] != y_predicted[i]:
            FP += 1
        if y_actual[i] == y_predicted[i] == 0:
            TN += 1
        if y_predicted[i] == 0 and y_actual[i] != y_predicted[i]:
            FN += 1

    return TP, FP, TN, FN

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def get_metrics(y_test, y_predicted):

    tp, fp, tn, fn = confusion_matrix(y_test, y_predicted)
    G_mean = np.sqrt((tp/(tp+fp)) * (tn/(tn+fp)))
    print('G-mean: %.4f' % G_mean)
    print('Balanced_Accuracy: %.4f' % metrics.balanced_accuracy_score(y_test, y_predicted))
    print('F1: %.4f' % metrics.f1_score(y_test, y_predicted, average="micro"))

def view_stats(new_df):
    rating_1 = len(new_df[new_df['Score'] == 1])
    rating_2 = len(new_df[new_df['Score'] == 2])
    rating_3 = len(new_df[new_df['Score'] == 3])
    rating_4 = len(new_df[new_df['Score'] == 4])
    rating_5 = len(new_df[new_df['Score'] == 5])

    labels = ['rating=1', 'rating=2', 'rating=3', 'rating=4', 'rating=5']
    reviews = [rating_1, rating_2, rating_3, rating_4, rating_5]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, reviews, width, label='Reviews')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Dataset after Removing duplicates')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)
    fig.tight_layout()
    plt.show()
    plt.savefig("reviews.png")
