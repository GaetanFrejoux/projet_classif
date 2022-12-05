# Authors : Frejoux Gaetan, Niord Mathieu

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from resources import (URL_TEST, EX1_RESULTS, K_RANGE, SAVE_FIG)
from utils import scatter_classif

# Known data
CLASSIF = [1]*50 + [2]*50 + [3]*50  # Training data classification

# Data loading
Data = loadmat(URL_TEST)  # Load data from the mat file
APPRENT = Data['test']
TEST = Data['x']
ORACLE = Data['clasapp']

def stats_reduce(data):
    return stats.mode(data, keepdims=True)[0][0]

def kppv(apprent, classe_origine, k, X):
    """
    Classification deduction function based on the K Nearest Neighbors algorithm

    Parameters
    ----------
    - apprent : Training data
    - classe_origine : Training data classification
    - k : Number of neighbors to consider
    - X : Data to classify

    Returns
    -------
    Array of estimated classification based on the average of the k nearest neighbors
    """
    res = []
    apprentT = np.transpose(apprent)  # Transpose the "apprent" matrix in order to calculate the norm after
    for i in range(len(X[0])):
        data_to_class = X[:, i] # Get each pair of data from X
        distances = np.linalg.norm(apprentT - data_to_class, axis=1)
        nearest_neighbor_ids = distances.argsort()[:k]
        nearest_neighbor_classes = [
            classe_origine[id] for id in nearest_neighbor_ids
        ] # [Map] Getting classes results from resulting ids
        res.append(stats_reduce(nearest_neighbor_classes))  # [Reduce] Getting the bigger occurence
    return res


def get_kppv(X_train, y_train, X_test, Oracle):
    min = 100
    classif = []
    for k in K_RANGE:
        Knn = kppv(X_train, y_train, k, X_test)
        Error_Percent = np.mean(Knn != Oracle) * 100
        if Error_Percent < min:
            min = Error_Percent 
            classif = Knn
    return classif

# Scatter graphic generator for data that are coming from a serialized file (.mat here)
def scatter_classif(Test_Mat, Classif_Array, save=False, name=''):
    plt.scatter(Test_Mat[:, 0], Test_Mat[:, 1], c=Classif_Array)
    print(Test_Mat[0, :])
    print(Test_Mat[1, :])
    plt.legend()
    if save and (not name == ''): plt.savefig(name + '_[CLASSIF].png')
    else: plt.show()
    plt.close()

def show_errors(Mat_Train, Classif_Train, Mat_Test, Oracle, Image_Name='', Save_Status=False):
    knn_errors = []
    # Run the knn with k included in K_RANGE
    for k in K_RANGE:
        knn_errors.append([k, np.mean(kppv(Mat_Train, Classif_Train, k, Mat_Test) != Oracle) * 100])
    # Construction
    plt.plot(np.array(knn_errors)[:, 0], np.array(knn_errors)[:, 1], label="KNN")
    plt.legend()
    # Save or display
    if Save_Status and (not Image_Name == ''): plt.savefig(Image_Name + '_[ERRORS].png')
    else: plt.show()
    plt.close() # Clear

def run_tests():
    print('\n=======\nResults for knn on ', K_RANGE, ' values\n')
    for k in K_RANGE:
        print('k =', k, '\t=>', np.mean(kppv(APPRENT, CLASSIF, k, TEST) != ORACLE) * 100, '%')
    print('=======\n')

Kppv = get_kppv(APPRENT, CLASSIF, TEST, ORACLE)
scatter_classif(TEST.T, ORACLE, SAVE_FIG, EX1_RESULTS + 'Classification')
scatter_classif(TEST.T, Kppv, SAVE_FIG, EX1_RESULTS + 'Kppv')
show_errors(APPRENT, CLASSIF, TEST, ORACLE, EX1_RESULTS + "Test", SAVE_FIG)
run_tests()