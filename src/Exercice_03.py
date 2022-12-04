# Authors : Frejoux Gaetan, Niord Mathieu

from pandas import read_excel
from resources import URL_WANG
from utils import run_knn_tests
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

TYPE = ["PHOG", "JCD", "CEDD", "FCTH", "FCH"]
EXCEL_WANG = read_excel(URL_WANG, sheet_name=[0, 1, 2, 3, 4], header=None)

for i in range(5):
    print(TYPE[i])
    CLASSIF_WANG = []
    for j in EXCEL_WANG[i][0]:
        CLASSIF_WANG.append(int(j.split('.')[0]) // 100)

    X_train, X_test, y_train, y_test = train_test_split(
        EXCEL_WANG[i].iloc[: , 1:], CLASSIF_WANG, test_size=0.2, random_state=42)

    run_knn_tests(X_train, y_train, X_test, y_test)
    print(1 - MultinomialNB().fit(X_train, y_train).score(X_test, y_test))

    CLASSIF_WANG.clear()
