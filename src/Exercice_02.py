from numpy import (mean, array)
from pandas import read_excel
from resources import (URL_KMEAN, URL_NON_GAUSSIEN, URL_PETIT, URL_GRAND)
from utils import (gnb, run_knn_tests)

# 2.1
EXCEL_1 = read_excel(URL_PETIT, sheet_name=[0, 1])
APPRENT_1 = EXCEL_1[0].iloc[:2, 1:].T
INCONNU_1 = EXCEL_1[1].iloc[:2, 1:].T
ORACLE_1 = EXCEL_1[1].iloc[2].values[1:]
CLASSIF_1 = [0]*20 + [1]*20 + [2]*20
print(mean(ORACLE_1 != gnb(APPRENT_1, CLASSIF_1, INCONNU_1)))
run_knn_tests(APPRENT_1, CLASSIF_1, INCONNU_1, ORACLE_1)

# 2.2
EXCEL_2 = read_excel(URL_GRAND, sheet_name=[0, 1])
APPRENT_2 = EXCEL_2[0].iloc[:2, 1:].T
INCONNU_2 = EXCEL_2[1].iloc[:2, 1:].T
ORACLE_2 = EXCEL_2[1].iloc[2].values[1:]
CLASSIF_2 = [0]*150 + [1]*150 + [2]*150
print(mean(ORACLE_2 != gnb(APPRENT_2, CLASSIF_2, INCONNU_2)))
run_knn_tests(APPRENT_2, CLASSIF_2, INCONNU_2, ORACLE_2)

# 2.3
EXCEL_3 = read_excel(URL_KMEAN, sheet_name=[0, 1])
APPRENT_3 = EXCEL_3[0].iloc[:2, 1:].T
INCONNU_3 = EXCEL_3[1].iloc[:2, 1:].T
ORACLE_3 = EXCEL_3[1].iloc[2].values[1:]
CLASSIF_3 = array(EXCEL_3[0].iloc[2].values[1:]).astype(int)
print(mean(ORACLE_3 != gnb(APPRENT_3, CLASSIF_3, INCONNU_3)))
run_knn_tests(APPRENT_3, CLASSIF_3, INCONNU_3, ORACLE_3)

# 2.4
EXCEL_4 = read_excel(URL_NON_GAUSSIEN, sheet_name=[0, 1])
APPRENT_4 = EXCEL_4[0].iloc[:2, 1:].T
INCONNU_4 = EXCEL_4[1].iloc[:2, 1:].T
ORACLE_4 = EXCEL_4[1].iloc[2].values[1:]
CLASSIF_4 = array(EXCEL_4[0].iloc[2].values[1:]).astype(int)
print(mean(ORACLE_4 != gnb(APPRENT_4, CLASSIF_4, INCONNU_4)))
run_knn_tests(APPRENT_4, CLASSIF_4, INCONNU_4, ORACLE_4)
