from numpy import (mean, array)
from pandas import read_excel
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from resources import (URL_PETIT, URL_GRAND)

# Known informations ###########################################################
CLASSIF_PETIT = [0]*20 + [1]*20 + [2]*20
CLASSIF_GRAND = [0]*150 + [1]*150 + [2]*150
MES1_LINE = 0
MES2_LINE = 1
ORACLE_LINE = 2
################################################################################

# Data loading ##################################################################
# p1_petit
EXCEL_PETIT = read_excel(URL_PETIT, sheet_name=[0, 1])
APPRENT_PETIT = array([EXCEL_PETIT[0].iloc[MES1_LINE].values[1:], EXCEL_PETIT[0].iloc[MES2_LINE].values[1:]]).T
INCONNU_PETIT = array([EXCEL_PETIT[1].iloc[MES1_LINE].values[1:], EXCEL_PETIT[1].iloc[MES2_LINE].values[1:]]).T
ORACLE_PETIT = EXCEL_PETIT[1].iloc[ORACLE_LINE].values[1:]
# p1_grand
EXCEL_GRAND = read_excel(URL_GRAND, sheet_name=[0, 1])
APPRENT_GRAND = array([EXCEL_GRAND[0].iloc[MES1_LINE].values[1:], EXCEL_GRAND[0].iloc[MES2_LINE].values[1:]]).T
INCONNU_GRAND = array([EXCEL_GRAND[1].iloc[MES1_LINE].values[1:], EXCEL_GRAND[1].iloc[MES2_LINE].values[1:]]).T
ORACLE_GRAND = EXCEL_GRAND[1].iloc[ORACLE_LINE].values[1:]
#################################################################################


def gnb(X_train, y_train, X_test):
    """
    Parametric discrimination - Gaussian Naive Bayes

    Refrences
    ---------
        - https://scikit-learn.org/stable/modules/classes.html?highlight=bayes#module-sklearn.naive_bayes
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
    naive_bayes = GaussianNB()
    return naive_bayes.fit(X_train, y_train).predict(X_test)


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
    neighbors = KNeighborsClassifier(n_neighbors = k)
    return neighbors.fit(X_train, y_train).predict(X_test)


def run_knn_tests(X_train, y_train, X_test, oracle):
    print('\n=======\nResults for knn on 1, 3, 5, 7, 9, 11, 13 and 15 values\n')
    for i in [1, 3, 5, 7, 13, 15]:
        print('k =', i, '\t=>', mean(oracle != knn(X_train, y_train, X_test, i) * 100), '%')
    print('=======\n')


print(mean(ORACLE_PETIT != gnb(APPRENT_PETIT, CLASSIF_PETIT, INCONNU_PETIT))* 100, "%")
run_knn_tests(APPRENT_PETIT, CLASSIF_PETIT, INCONNU_PETIT, ORACLE_PETIT)

print(mean(ORACLE_GRAND != gnb(APPRENT_GRAND, CLASSIF_GRAND, INCONNU_GRAND))* 100, "%")
run_knn_tests(APPRENT_GRAND, CLASSIF_GRAND, INCONNU_GRAND, ORACLE_GRAND)