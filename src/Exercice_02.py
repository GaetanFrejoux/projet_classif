from pandas import read_excel
from resources import (URL_KMEAN, URL_NON_GAUSSIEN, URL_PETIT, URL_GRAND, EX2_RESULTS)
from utils import (gnb, show_errors, scatter_classif, get_knn, run_knn_tests)

def show_classif_results(X_test, Knn, Gnb, Oracle, Image_Name='', Save_Status=False):
    """
    Will construct and save/display the classification results in a scatter graphic format,
    in the same time for the Knn, the Gnb and the Oracle.

    Parameters
    ----------
        - X_test: The data that need to be classified,
        - Knn: The returned classification result from the knn function (),
        - Gnb: The returned classification result from the gnn function (),
        - Oracle: The expected classification result,
        - Image_Name: The name of the saved image,
        - Save_Status: The saving behaviour (does it save or does it display ?). False as default.
    """
    scatter_classif(X_test, Knn, Save_Status, EX2_RESULTS + Image_Name + '_KNN')
    scatter_classif(X_test, Gnb, Save_Status, EX2_RESULTS + Image_Name + '_GNB')
    scatter_classif(X_test, Oracle, Save_Status, EX2_RESULTS + Image_Name + '_ORACLE')

# 2.1 - Little array
################################################################################
EXCEL_1 = read_excel(URL_PETIT, sheet_name=[0, 1])

APPRENT_1 = EXCEL_1[0].iloc[:2, 1:].T
INCONNU_1 = EXCEL_1[1].iloc[:2, 1:].T
ORACLE_1 = EXCEL_1[1].iloc[2].values[1:]

CLASSIF_1 = [0]*20 + [1]*20 + [2]*20

Knn_Petit = get_knn(APPRENT_1, CLASSIF_1, INCONNU_1, ORACLE_1)
Gnb_Petit = gnb(APPRENT_1, CLASSIF_1, INCONNU_1)

run_knn_tests(APPRENT_1, CLASSIF_1, INCONNU_1, ORACLE_1)
show_errors(APPRENT_1, CLASSIF_1, INCONNU_1, ORACLE_1, EX2_RESULTS + 'PETIT', True)
show_classif_results(INCONNU_1, Knn_Petit, Gnb_Petit, ORACLE_1, 'PETIT', True)
################################################################################

# 2.2 - Big array
################################################################################
EXCEL_2 = read_excel(URL_GRAND, sheet_name=[0, 1])

APPRENT_2 = EXCEL_2[0].iloc[:2, 1:].T
INCONNU_2 = EXCEL_2[1].iloc[:2, 1:].T
ORACLE_2 = EXCEL_2[1].iloc[2].values[1:]

CLASSIF_2 = [0]*150 + [1]*150 + [2]*150

Knn_Grand = get_knn(APPRENT_2, CLASSIF_2, INCONNU_2, ORACLE_2)
Gnb_Grand = gnb(APPRENT_2, CLASSIF_2, INCONNU_2)

run_knn_tests(APPRENT_2, CLASSIF_2, INCONNU_2, ORACLE_2)
show_errors(APPRENT_2, CLASSIF_2, INCONNU_2, ORACLE_2, EX2_RESULTS + 'GRAND', True)
show_classif_results(INCONNU_2, Knn_Grand, Gnb_Grand, ORACLE_2, 'GRAND', True)
################################################################################

# 2.3 - K-Mean
################################################################################
EXCEL_3 = read_excel(URL_KMEAN, sheet_name=[0, 1])

APPRENT_3 = EXCEL_3[0].iloc[:2, 1:].T
INCONNU_3 = EXCEL_3[1].iloc[:2, 1:].T
ORACLE_3 = EXCEL_3[1].iloc[2].values[1:]

CLASSIF_3 = EXCEL_3[0].iloc[2].values[1:].astype(int)

Knn_KMean = get_knn(APPRENT_3, CLASSIF_3, INCONNU_3, ORACLE_3)
Gnb_KMean = gnb(APPRENT_3, CLASSIF_3, INCONNU_3)

run_knn_tests(APPRENT_3, CLASSIF_3, INCONNU_3, ORACLE_3)
show_errors(APPRENT_3, CLASSIF_3, INCONNU_3, ORACLE_3, EX2_RESULTS + 'KMEAN', True)
show_classif_results(INCONNU_3, Knn_KMean, Gnb_KMean, ORACLE_3, 'KMEAN', True)
################################################################################

# 2.4 - Non Gaussian
################################################################################
EXCEL_4 = read_excel(URL_NON_GAUSSIEN, sheet_name=[0, 1])

APPRENT_4 = EXCEL_4[0].iloc[:2, 1:].T
INCONNU_4 = EXCEL_4[1].iloc[:2, 1:].T
ORACLE_4 = EXCEL_4[1].iloc[2].values[1:]

CLASSIF_4 = EXCEL_4[0].iloc[2].values[1:].astype(int)

Knn_Non_Gaussian = get_knn(APPRENT_4, CLASSIF_4, INCONNU_4, ORACLE_4)
Gnb_Non_Gaussian = gnb(APPRENT_4, CLASSIF_4, INCONNU_4)

run_knn_tests(APPRENT_4, CLASSIF_4, INCONNU_4, ORACLE_4)
show_errors(APPRENT_4, CLASSIF_4, INCONNU_4, ORACLE_4, EX2_RESULTS + 'NON_GAUSSIEN', True)
show_classif_results(INCONNU_4, Knn_Non_Gaussian, Gnb_Non_Gaussian, ORACLE_4, 'NON_GAUSSIEN', True)
################################################################################