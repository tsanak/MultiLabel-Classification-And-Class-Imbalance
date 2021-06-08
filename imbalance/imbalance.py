import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import keras.metrics as k_metrics
import preprocess as pr


def run_bilstm(X_train, X_test, y_train, y_test):

    tokenizer = Tokenizer(num_words=25000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1


    maxlen = 200

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
    x = Bidirectional(LSTM(64, dropout=0.3))(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(learning_rate=0.001)

    METRICS = [
        k_metrics.TruePositives(name='tp'),
        k_metrics.FalsePositives(name='fp'),
        k_metrics.TrueNegatives(name='tn'),
        k_metrics.FalseNegatives(name='fn'),
        k_metrics.BinaryAccuracy(name='accuracy'),
        k_metrics.Precision(name='precision'),
        k_metrics.Recall(name='recall'),
        k_metrics.AUC(name='auc'),
        k_metrics.AUC(name='prc', curve='PR'),
        'acc'
    ]

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])

    print(model.summary())
    # batch_size: 128
    history = model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1, validation_split=0.2)
    # score = model.evaluate(X_test, y_test, verbose=1)
    y_predicted_nn = model.predict(X_test)

    y_predicted = []
    for prediction in y_predicted_nn:
        result = np.argmax(prediction)
        y_predicted.append(result)

    pr.get_metrics(y_test, y_predicted)

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


if __name__ == "__main__":
    print("Imbalanced project")

    df = pr.preprocess()
    new_df = df.drop_duplicates(subset=['Text'])
    print(len(new_df))

    # for bar char stats
    pr.view_stats(new_df)


    X_train, X_test, y_train, y_test = pr.sklearn_train_test(new_df)

    tfidf_x_train, tfidf_x_test = pr.tf_idf(X_train, X_test)

    method = 'Smote'
    if method == 'Smote':
        # Smote
        X_train, y_train = pr.Smote(tfidf_x_train, y_train)
    elif method == 'Near_Miss':
        # Near_Miss
        X_train, y_train = pr.Near_miss(tfidf_x_train, y_train)
    elif method == 'SmoteTomek':
        # Smote - Tomek
        X_train, y_train = pr.Smotetomek(tfidf_x_train, y_train)
    else:
        X_train, X_test, y_train, y_test = pr.sklearn_train_test(new_df, 'undersample')
        tfidf_x_train, tfidf_x_test = pr.tf_idf(X_train, X_test)
        X_train = tfidf_x_train

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(tfidf_x_test)

    pr.get_metrics(y_test, y_predicted)

    # Multinomial Naive Bayes

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(tfidf_x_test)
    pr.get_metrics(y_test, y_predicted)

    # Logistic Regression

    logreg = LogisticRegression(C=1e5, solver='lbfgs', max_iter=5000, multi_class='multinomial')
    logreg.fit(X_train, y_train)
    y_predicted = logreg.predict(tfidf_x_test)
    pr.get_metrics(y_test, y_predicted)


    run_bilstm(X_train, X_test, y_train, y_test)
