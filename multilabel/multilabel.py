from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.ensemble import RakelD, RakelO
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from skmultilearn.utils import measure_per_label
from scipy.sparse import csr_matrix, lil_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
import keras.backend as K
from keras.metrics import BinaryAccuracy

import preprocess as pr


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def run_bilstm(X_train, X_test, y_train, y_test):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100
    number_of_labels = 20

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embeddings_dictionary = dict()

    embedding_dimensions = 300
    glove_file = open('../glove.6B.' + str(embedding_dimensions) + 'd.txt', encoding="utf8")
    print('start reading glove')

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    print('finished reading glove')

    embedding_matrix = np.zeros((vocab_size, embedding_dimensions))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, embedding_dimensions, weights=[embedding_matrix], trainable=False)(inputs)

    x = Bidirectional(LSTM(128, dropout=0.3))(x)
    outputs = Dense(number_of_labels, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc', get_f1])
    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=128, epochs=25, verbose=1, validation_split=0.2)
    preds = model.predict(X_test)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    acc = metrics.accuracy_score(y_test, preds)
    print('ACC: %.3f' % acc)

    prec = metrics.precision_score(y_test, preds, average="micro")
    print('PREC: %.3f' % prec)

    rec = metrics.recall_score(y_test, preds, average="micro")
    print('REC: %.3f' % rec)

    f1 = metrics.f1_score(y_test, preds, average="micro")
    print('F1: %.3f' % f1)
    hamming_loss = metrics.hamming_loss(y_test, preds)
    print('HL: %.5f' % hamming_loss)

    print('MACRO')
    prec = metrics.precision_score(y_test, preds, average="macro")
    print('PREC: %.3f' % prec)

    rec = metrics.recall_score(y_test, preds, average="macro")
    print('REC: %.3f' % rec)

    f1 = metrics.f1_score(y_test, preds, average="macro")
    print('F1: %.3f' % f1)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('################')
    print('F1 per label')
    print(metrics.f1_score(y_test, preds, average=None))
    print('################')


def run_predict(ml_classifier, x_train, y_train, x_test, y_test):
    ml_classifier.fit(x_train, y_train)
    y_predicted = ml_classifier.predict(x_test)

    f1 = metrics.f1_score(y_test, y_predicted, average="micro")
    f1_no_avg = metrics.f1_score(y_test, y_predicted, average=None)
    f1_macro = metrics.f1_score(y_test, y_predicted, average="macro")
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average="micro")
    precision = metrics.precision_score(y_test, y_predicted, average="micro")
    hamming_loss = metrics.hamming_loss(y_test, y_predicted)

    return {
        'predictions': y_predicted,
        'metrics': {
            'f1': f1,
            'f1_no_avg': f1_no_avg,
            'f1_macro': f1_macro,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'hamming_loss': hamming_loss,
        }
    }


def print_metrics(metric):
    print('Accuracy: %2f' % metric['accuracy'])
    print('Precision: %2f' % metric['precision'])
    print('Recall: %2f' % metric['recall'])
    print('F1: %2f' % metric['f1'])
    print(metric['f1_no_avg'])
    print('Macro F1: %2f' % metric['f1_macro'])
    print('Hamming loss: %2f' % metric['hamming_loss'])


if __name__ == "__main__":
    print("MultiLabel project")

    df = pr.preprocess()

    X_train, X_test, y_train, y_test = pr.sklearn_train_test(df)

    run_bilstm(X_train, X_test, y_train, y_test)

    tfidf_x_train, tfidf_x_test = pr.tf_idf(X_train, X_test)

    lil_x_train = lil_matrix(tfidf_x_train).toarray()
    lil_y_train = lil_matrix(y_train).toarray()
    lil_x_test = lil_matrix(tfidf_x_test).toarray()

    # ------------------BinaryRelevance---------------------
    classifier = BinaryRelevance(
        classifier=MultinomialNB(),
        require_dense=[False, True]
    )

    output = run_predict(classifier, tfidf_x_train, y_train, tfidf_x_test, y_test)
    print('MultinomialNB | BINARY RELEVANCE')
    print_metrics(output['metrics'])
    print('================================')
    # ------------------ClassifierChain---------------------
    classifier = ClassifierChain(
        classifier=MultinomialNB(),
        require_dense=[False, True]
    )
    output = run_predict(classifier, tfidf_x_train, y_train, tfidf_x_test, y_test)
    print('MultinomialNB | ClassifierChain')
    print_metrics(output['metrics'])
    print('================================')

    # ------------------LabelPowerSet---------------------
    classifier = LabelPowerset(
        classifier=MultinomialNB(),
        require_dense=[False, True]
    )
    output = run_predict(classifier, tfidf_x_train, y_train, tfidf_x_test, y_test)
    print('MultinomialNB | LabelPowerset')
    print_metrics(output['metrics'])
    print('================================')


    # ------------------MLkNN---------------------
    classifier = MLkNN(k=10)
    output = run_predict(classifier, lil_x_train, lil_y_train, lil_x_test, y_test)
    print('MLkNN')
    print_metrics(output['metrics'])
    print('================================')

