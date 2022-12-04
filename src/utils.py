# Authors : Frejoux Gaetan, Niord Mathieu

from numpy import (mean, array)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from resources import K_RANGE

def get_knn(X_train, y_train, X_test, Oracle):
    min = 100
    knn_result = []
    for k in K_RANGE:
        Knn = knn(X_train, y_train, X_test, k)
        Error_Percent = mean(Knn != Oracle) * 100
        if Error_Percent < min:
            min = Error_Percent 
            knn_result = Knn
    return knn_result


def gnb(X_train, y_train, X_test):
    """
    Parametric discrimination - Gaussian Naive Bayes

    References
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

    Reference
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


def run_knn_tests(X_train, y_train, X_test, Oracle):
    print('\n=======\nResults for knn on 1, 3, 5, 7, 9, 11, 13 and 15 values\n')
    for k in K_RANGE:
        print('k =', k, '\t=>', mean(knn(X_train, y_train, X_test, k) != Oracle))
    print('=======\n')

# 
def show_errors(X_train, y_train, X_test, Oracle, Image_Name='', Save_Status=False):
    knn_errors = []
    gnn_error = mean(Oracle != gnb(X_train, y_train, X_test)) * 100
    # Run the knn with k included in K_RANGE
    for k in K_RANGE:
        knn_errors.append([k, mean(knn(X_train, y_train, X_test, k) != Oracle) * 100])
    # Construction
    plt.plot(array(knn_errors)[:, 0], array(knn_errors)[:, 1], label="KNN")
    plt.plot([1, 15], [gnn_error, gnn_error], label="GNB")
    # Save or display
    if Save_Status and (not Image_Name == ''): plt.savefig(Image_Name + '_[ERRORS].png')
    else: plt.show()
    plt.close() # Clear

# Scatter graphic generator for data that are coming from an excel file
def scatter_classif(X_test, Classif_Array, save=False, name=''):
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=Classif_Array)
    if save and (not name == ''): plt.savefig(name + '_[CLASSIF].png')
    else: plt.show()
    plt.close()
