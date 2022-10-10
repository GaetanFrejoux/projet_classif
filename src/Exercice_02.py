from numpy import (mean, array)
from pandas import read_excel, ExcelFile
from resources import (URL_KMEAN, URL_NON_GAUSSIEN,
                       URL_PETIT, URL_GRAND)
from utils import (gnb, knn, run_knn_tests)

# Known informations ###########################################################
CLASSIF_PETIT = [0]*20 + [1]*20 + [2]*20
CLASSIF_GRAND = [0]*150 + [1]*150 + [2]*150
MES1_LINE = 0
MES2_LINE = 1
ORACLE_LINE = 2
################################################################################

# Data loading ##################################################################

# p1_petit
EXCEL_PETIT = read_excel(URL_PETIT, sheet_name=[0, 1])
APPRENT_PETIT = array([EXCEL_PETIT[0].iloc[MES1_LINE].values[1:],
                      EXCEL_PETIT[0].iloc[MES2_LINE].values[1:]]).T
INCONNU_PETIT = array([EXCEL_PETIT[1].iloc[MES1_LINE].values[1:],
                      EXCEL_PETIT[1].iloc[MES2_LINE].values[1:]]).T
ORACLE_PETIT = EXCEL_PETIT[1].iloc[ORACLE_LINE].values[1:]

# p1_grand
EXCEL_GRAND = read_excel(URL_GRAND, sheet_name=[0, 1])
APPRENT_GRAND = array([EXCEL_GRAND[0].iloc[MES1_LINE].values[1:],
                      EXCEL_GRAND[0].iloc[MES2_LINE].values[1:]]).T
INCONNU_GRAND = array([EXCEL_GRAND[1].iloc[MES1_LINE].values[1:],
                      EXCEL_GRAND[1].iloc[MES2_LINE].values[1:]]).T
ORACLE_GRAND = EXCEL_GRAND[1].iloc[ORACLE_LINE].values[1:]

# p1_Kmeans
EXCEL_KMEANS = read_excel(URL_KMEAN, sheet_name=[0, 1])
APPRENT_KMEANS = array([EXCEL_KMEANS[0].iloc[MES1_LINE].values[1:],
                       EXCEL_KMEANS[0].iloc[MES2_LINE].values[1:]]).T
CLASSIF_KMEANS = array(EXCEL_KMEANS[0].iloc[2].values[1:]).astype(int)
INCONNU_KMEANS = array([EXCEL_KMEANS[1].iloc[MES1_LINE].values[1:],
                       EXCEL_KMEANS[1].iloc[MES2_LINE].values[1:]]).T
ORACLE_KMEANS = EXCEL_KMEANS[1].iloc[ORACLE_LINE].values[1:]

# p1_NonGaussien
EXCEL_NONGAUSSIEN = read_excel(URL_NON_GAUSSIEN, sheet_name=[0, 1])
APPRENT_NONGAUSSIEN = array([EXCEL_NONGAUSSIEN[0].iloc[MES1_LINE].values[1:],
                            EXCEL_NONGAUSSIEN[0].iloc[MES2_LINE].values[1:]]).T
CLASSIF_NONGAUSSIEN = array(
    EXCEL_NONGAUSSIEN[0].iloc[2].values[1:]).astype(int)
INCONNU_NONGAUSSIEN = array([EXCEL_NONGAUSSIEN[1].iloc[MES1_LINE].values[1:],
                            EXCEL_NONGAUSSIEN[1].iloc[MES2_LINE].values[1:]]).T
ORACLE_NONGAUSSIEN = EXCEL_NONGAUSSIEN[1].iloc[ORACLE_LINE].values[1:]

# 2.1
print(mean(ORACLE_PETIT != gnb(APPRENT_PETIT, CLASSIF_PETIT, INCONNU_PETIT)))
run_knn_tests(APPRENT_PETIT, CLASSIF_PETIT, INCONNU_PETIT, ORACLE_PETIT)

# 2.2
print(mean(ORACLE_GRAND != gnb(APPRENT_GRAND, CLASSIF_GRAND, INCONNU_GRAND)))
run_knn_tests(APPRENT_GRAND, CLASSIF_GRAND, INCONNU_GRAND, ORACLE_GRAND)

# 2.3
print(mean(ORACLE_KMEANS != gnb(APPRENT_KMEANS, CLASSIF_KMEANS, INCONNU_KMEANS)))
run_knn_tests(APPRENT_KMEANS, CLASSIF_KMEANS, INCONNU_KMEANS, ORACLE_KMEANS)

# 2.4
print(mean(ORACLE_NONGAUSSIEN != gnb(APPRENT_NONGAUSSIEN,
      CLASSIF_NONGAUSSIEN, INCONNU_NONGAUSSIEN)))
run_knn_tests(APPRENT_NONGAUSSIEN, CLASSIF_NONGAUSSIEN,
              INCONNU_NONGAUSSIEN, ORACLE_NONGAUSSIEN)