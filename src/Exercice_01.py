import numpy as np
from scipy.io import loadmat
from utils import *

### Initialisation des données
# Url de la donnée
url = "./res/p1_test.mat"
# Chargement du fichier .mat sérialisé afin de récupérer des données
# d'apprentissage, résultats et des données à classifier
Data = loadmat(url)
# Classification des données d'apprentissage
classification_exo1 = [1]*50 + [2]*50 + [3]*50

# Fonction de classification de données se basant sur le modèle KPPV
# Parametres :
#   - apprent : 
#   - classe_origine :
#   - k :
#   - X :
# Retour : Tableau des classifications estimées tq la classe de x vaut res[x]
def kppv(apprent, classe_origine, k, X):
  res = []
  apprentT = np.transpose(apprent)                                # Transpose the "apprent" matrix in order to calculate the norm after
  for i in range(len(X[0])):
    data_to_class = X[:, i]                                       # Get each pair of data from X
    distances = np.linalg.norm(apprentT - data_to_class, axis=1)
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_classes = [
      classe_origine[id] for id in nearest_neighbor_ids
    ]                                                             # [Map] Getting classes results from resulting ids
    res.append(reduce(nearest_neighbor_classes))                  # [Reduce] Getting the bigger occurence
  return res