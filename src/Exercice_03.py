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


# function that asks the user to enter a number between 0 and 999 for the image to be classified
# and ask the user to enter a number between 1 and 5 for the descriptor to be used
# and ask ther user 1 for the KNN algorithm and 2 for the Naive Bayes algorithm
# and print the classification of the image
def classif_image_user_input():
    image_number = int(input("Enter a number between 0 and 999 for the image to be classified: "))
    descriptor_number = int(input("Enter a number between 1 and 5 for the descriptor to be used: PHOG = 1, JCD = 2, CEDD = 3, FCTH = 4, FCH = 5: "))
    algorithm_number = int(input("Enter 1 for the KNN algorithm and 2 for the Naive Bayes algorithm: "))
    image_name = str(image_number) + ".jpg"
    image_class = int(image_name.split('.')[0]) // 100
    descriptor_name = TYPE[descriptor_number - 1]
    descriptor = EXCEL_WANG[descriptor_number - 1].iloc[image_number, 1:]
    descriptor_list = EXCEL_WANG[descriptor_number - 1].iloc[:, 1:]
    class_list = []
    for j in EXCEL_WANG[descriptor_number - 1][0]:
        class_list.append(int(j.split('.')[0]) // 100)
    X_train, X_test, y_train, y_test = train_test_split(
        descriptor_list, class_list, test_size=0.2, random_state=100)

    if algorithm_number == 1:
        classification = run_knn_tests(X_train, y_train, descriptor, image_class)
    elif algorithm_number == 2:
        classification = MultinomialNB().fit(X_train, y_train).predict([descriptor])

    print("The classification of the image is: " + str(classification))


classif_image_user_input()
