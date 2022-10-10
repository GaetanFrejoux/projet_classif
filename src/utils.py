from numpy import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


def stats_reduce(data):
    return stats.mode(data, keepdims=True)[0][0]


def gnb(X_train, y_train, X_test):
    """
    Parametric discrimination - Gaussian Naive Bayes

    Refrences
    ---------
        #module-sklearn.naive_bayes
        - https://scikit-learn.org/stable/modules/classes.html?highlight=bayes
        - https://scikit-learn.org/stable/modules/naive_bayes.html?highlight=bayes

    Parameters
    ----------
        - X_train: Training data,
        - y_train: Training data classification,
        - X_test: Data to classify

    Returns
    -------
    TODO
    """
    return GaussianNB().fit(X_train, y_train).predict(X_test)


def knn(X_train, y_train, X_test, k):
    """
    K Nearest Neighbors

    Refrence
    --------
    https://scikit-learn.org/stable/modules/neighbors.html

    Parameters
    ----------
        - X_train: ,
        - y_train: ,
        - X_test: ,
        - k: Number of neighbors to use by default for kneighbors queries

    Returns
    -------
    TODO
    """
    return KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)


def run_knn_tests(X_train, y_train, X_test, oracle):
    print('\n=======\nResults for knn on 1, 3, 5, 7, 9, 11, 13 and 15 values\n')
    for i in [1, 3, 5, 7, 13, 15]:
        print('k =', i, '\t=>', mean(knn(X_train, y_train, X_test, i) != oracle))
    print('=======\n')
