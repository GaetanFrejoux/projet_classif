import numpy as np
from pandas.util._decorators import (Appender, doc)
from scipy.io import loadmat
from resources import URL_TEST
from utils import stats_reduce

# Known data
CLASSIF = [1]*50 + [2]*50 + [3]*50  # Training data classification

# Data loading
Data = loadmat(URL_TEST)  # Load data from the mat file


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
  apprentT = np.transpose(apprent)                                # Transpose the "apprent" matrix in order to calculate the norm after
  for i in range(len(X[0])):
    data_to_class = X[:, i]                                       # Get each pair of data from X
    distances = np.linalg.norm(apprentT - data_to_class, axis=1)
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_classes = [
      classe_origine[id] for id in nearest_neighbor_ids
    ]                                                             # [Map] Getting classes results from resulting ids
    res.append(stats_reduce(nearest_neighbor_classes))            # [Reduce] Getting the bigger occurence
  return res


def run_tests():
    print('\n=======\nResults for knn on 1, 3, 5, 7, 9, 11, 13 and 15 values\n')
    for i in [1, 3, 5, 7, 13, 15]:
        print('k =', i, '\t=>',  np.mean(kppv(Data['test'], CLASSIF, i, Data['x']) != Data['clasapp']) * 100, '%')
    print('=======\n')

run_tests()