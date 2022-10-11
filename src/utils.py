from numpy import (mean, array)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from matplotlib import pyplot as plt

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


def show_errors(X_train, y_train, X_test, oracle, save=False, name='errors'):
    knn_errors = []
    gnn_error = mean(oracle != gnb(X_train, y_train, X_test)) * 100
    for k in [1, 3, 5, 7, 13, 15]:
        knn_errors.append([k, mean(knn(X_train, y_train, X_test, k) != oracle) * 100])
    plt.plot(array(knn_errors)[:, 0], array(knn_errors)[:, 1], label="KNN")
    plt.plot([1, 15], [gnn_error, gnn_error], label="GNB")
    plt.legend()
    if save: plt.savefig(name + '_ERRORS.png')
    else: plt.show()
    plt.close()

def scatter_classif(X_test, classif_array, save=False, name='classif'):
    #scatter based on X_test and classed by knn_classif
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=classif_array)
    plt.legend()
    if save: plt.savefig(name + '_CLASSIF.png')
    else: plt.show()
    plt.close()

def show_classif(X_test, knn_classif, gnn_classif, oracle, save=False, name='classif'):
    #scatter based on X_test and classed by knn_classif
    scatter_classif(X_test, knn_classif, save, name + '_KNN')
    #scatter based on X_test and classed by gnn_classif
    scatter_classif(X_test, gnn_classif, save, name + '_GNB')
    #scatter based on X_test and classed by oracle
    scatter_classif(X_test, oracle, save, name + '_ORACLE')

