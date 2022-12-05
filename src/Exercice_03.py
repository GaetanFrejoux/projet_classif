# Authors : Frejoux Gaetan, Niord Mathieu

import pandas as pd
from resources import URL_WANG
from utils import run_knn_tests
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

TYPE = ["PHOG", "JCD", "CEDD", "FCTH", "FCH"]
EXCEL_WANG = pd.read_excel(URL_WANG, sheet_name=[0, 1, 2, 3, 4], header=None)

# each descriptor
for i in range(5):
    print(TYPE[i])
    CLASSIF_WANG = []
    for j in EXCEL_WANG[i][0]:
        CLASSIF_WANG.append(int(j.split('.')[0]) // 100)

    X_train, X_test, y_train, y_test = train_test_split(
        EXCEL_WANG[i].iloc[:, 1:], CLASSIF_WANG, test_size=0.2, random_state=100)

    run_knn_tests(X_train, y_train, X_test, y_test)
    print(1 - MultinomialNB().fit(X_train, y_train).score(X_test, y_test))

    # confusion matrix
    y_pred = MultinomialNB().fit(X_train, y_train).predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    CLASSIF_WANG.clear()

# all descriptors in one
print("all")
CLASSIF_WANG = []
for j in EXCEL_WANG[0][0]:
    CLASSIF_WANG.append(int(j.split('.')[0]) // 100)

EXCEL_WANG_ALL = pd.concat([EXCEL_WANG[0].iloc[:, 1:],
                            EXCEL_WANG[1].iloc[:, 1:],
                            EXCEL_WANG[2].iloc[:, 1:],
                            EXCEL_WANG[3].iloc[:, 1:],
                            EXCEL_WANG[4].iloc[:, 1:]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    EXCEL_WANG_ALL, CLASSIF_WANG, test_size=0.2, random_state=100)

run_knn_tests(X_train, y_train, X_test, y_test)
print(1 - MultinomialNB().fit(X_train, y_train).score(X_test, y_test))

# confusion matrix
y_pred = MultinomialNB().fit(X_train, y_train).predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()