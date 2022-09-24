import numpy as np
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

url = "./res/p1_petit.xlsx"
excel_data = pd.read_excel(url, sheet_name=[0, 1])

# See https://scikit-learn.org/stable/modules/classes.html?highlight=bayes#module-sklearn.naive_bayes for help on Naive Bayes
# Also see https://scikit-learn.org/stable/modules/naive_bayes.html?highlight=bayes

apprent = np.array([excel_data[0].iloc[0].values[1:], excel_data[0].iloc[1].values[1:]]).T
test    = np.array([excel_data[1].iloc[0].values[1:], excel_data[1].iloc[1].values[1:]]).T
oracle  = excel_data[1].iloc[2].values[1:]

classifications = [0]*20 + [1]*20 + [2]*20

def gnb(X_train, y_train, X_test):
    gnb = GaussianNB()
    return gnb.fit(X_train, y_train).predict(X_test)

def knn(X_train, y_train, X_test, k):
    neigh = KNeighborsClassifier(n_neighbors = k)
    return neigh.fit(X_train, y_train).predict(X_test)

print(np.mean(oracle != gnb(apprent, classifications, test))* 100, "%")
print(np.mean(oracle != knn(apprent, classifications, test, 3))* 100, "%")