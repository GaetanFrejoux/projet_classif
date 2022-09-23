import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import stats

url = "./res/p1_petit.xlsx"
excel_data = pd.read_excel(url, sheet_name=[0, 1])

mes1_app = excel_data[0].iloc[0].values
mes2_app = excel_data[0].iloc[1].values
classifications = [0]*20 + [1]*20 + [2]*20

mes1 = excel_data[1].iloc[0].values
mes2 = excel_data[1].iloc[1].values
oracle = excel_data[1].iloc[2].values

# See https://scikit-learn.org/stable/modules/classes.html?highlight=bayes#module-sklearn.naive_bayes for help on Naive Bayes
# Also see https://scikit-learn.org/stable/modules/naive_bayes.html?highlight=bayes

print(mes1)