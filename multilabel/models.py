from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
import multilabel as ml


if __name__ == "__main__":
    print("MultiLabel project")

    df = ml.preprocess()

    X_train, X_test, y_train, y_test = ml.sklearn_train_test(df)

    bow_x_train, bow_x_test = ml.bag_of_words(X_train, X_test)
    tfidf_x_train, tfidf_x_test = ml.tf_idf(X_train, X_test)


    # ------------------BinaryRelevance---------------------

    # classifier = BinaryRelevance(
    #     classifier=MultinomialNB(),
    #     require_dense=[False, True]
    # )

    # classifier = BinaryRelevance(
    #     classifier=SVC(kernel="poly"),
    #     require_dense=[False, True]
    # )

    # parameters = [
    #     {
    #         'classifier': [MultinomialNB()],
    #         'classifier__alpha': [0.7, 1.0],
    #     },
    #     {
    #         'classifier': [SVC()],
    #         'classifier__kernel': ['rbf', 'linear'],
    #     },
    # ]
    #
    # clf = GridSearchCV(BinaryRelevance(), parameters)

    # ------------------ClassifierChain---------------------
    # classifier = ClassifierChain(
    #     classifier=MultinomialNB(),
    #     require_dense=[False, True]
    # )

    # ------------------ClassifierChain---------------------
    classifier = LabelPowerset(
        classifier=MultinomialNB(),
        require_dense=[False, True]
    )

    # train
    # classifier.fit(bow_x_train, y_train)
    classifier.fit(tfidf_x_train, y_train)

    # predict
    # y_predicted = classifier.predict(bow_x_test)
    y_predicted = classifier.predict(tfidf_x_test)

    f1 = metrics.f1_score(y_test, y_predicted, average="micro")
    # accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average="micro")
    precision = metrics.precision_score(y_test, y_predicted, average="micro")

    # print('Accuracy: %2f' % accuracy)
    print('Precision: %2f' % precision)
    print('Recall: %2f' % recall)
    print('F1: %2f' % f1)
