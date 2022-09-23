from Exercice_01 import *

### Tests Exercice_01
def test_01():
    print("\n=======\nExercice 01 : Résultats du kppv sur les valeurs 1, 3, 5, 7, 9, 11, 13, 15\n")
    for i in [1, 3, 5, 7, 13, 15]:
        print("k =", i, "\t=>",  np.mean(kppv(Data["test"], classification_exo1, i, Data["x"]) != Data["clasapp"]) * 100, "%")
    print("=======\n")
test_01()

## Tests Exercice_02
def test_02_1():
    print("\n=======\nExercice 02\n")
    print("=======\n")
test_02_1()