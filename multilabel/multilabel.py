import json
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

data = pd.read_csv('dataset/full_movies_metadata.csv', sep=',')

# Remove rows with no labels and find unique labels
rowsWithNoLabel = []
genreList = []
for index, g in enumerate(data['genres']):
    fix_json = g.replace("\'", "\"")
    genre = json.loads(fix_json)
    if len(genre) == 0:
        rowsWithNoLabel.append(index)
    for current_genre in genre:
        if current_genre['name'] not in genreList:
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

data = {
    'overview': data['overview']
}

for g in genreDict:
    data[g] = np.array(genreDict[g])

# Create DataFrame
data = pd.DataFrame(data)
X = data['overview']
y = np.asarray(data[data.columns[1:]])

# data.drop(['genres', 'adult', 'belongs_to_collection', 'budget', 'homepage', 'id', 'original_language',
#            'original_title', 'popularity', 'poster_path', 'production_companies', 'production_countries',
#            'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'video', 'vote_average',
#            'vote_count'
#            ], inplace=True, axis=1)

# y = y.values.reshape(-1, 1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, test_size=0.2)

vectorizer = CountVectorizer(ngram_range=(1, 2))

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
classifier = BinaryRelevance(
    classifier=SVC(),
    require_dense=[False, True]
)
# train
classifier.fit(x_train, y_train)

# predict
y_predicted = classifier.predict(x_test)

f1 = metrics.f1_score(y_test, y_predicted, average="macro")
accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted, average="macro")
precision = metrics.precision_score(y_test, y_predicted, average="macro")

print('Accuracy: %2f' % accuracy)
print('Precision: %2f' % precision)
print('Recall: %2f' % recall)
print('F1: %2f' % f1)
