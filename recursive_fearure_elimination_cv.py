# -*- coding: utf-8 -*-
# @Time    : 5/2/18 6:02 PM
# @Author  : Jason Lin
# @File    : recursive_fearure_elimination_cv.py
# @Software: PyCharm


import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib
import numpy as np
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def frank_on_data1():
    cancer_detection_data = pd.read_csv("data/Student_CancerDetection.csv")
    data_arr = np.array(cancer_detection_data)
    feature_name = cancer_detection_data.columns
    print(feature_name)
    X = data_arr[:, :9]
    y_label = data_arr[:, 9]
    # estimator = SVC(kernel="linear")
    estimator = ExtraTreesClassifier(n_estimators=300)
    selector = RFE(estimator, n_features_to_select =1, verbose=1)
    selector = selector.fit(X, y_label)

    print(feature_name[selector.ranking_ - 1])
    print(selector.ranking_)

    p = matplotlib.rcParams
    p["figure.subplot.bottom"] = 0.2

    plt.figure()
    plt.title("Feature importances")

    plt.bar(range(9), range(9, 0,-1) ,
            color="r", align="center")

    plt.xticks(range(9), feature_name[selector.ranking_ - 1], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def frank_on_data2():
    cancer_detection_data = pd.read_csv("data/Student_DifferentCancerDetection.csv")
    data_arr = np.array(cancer_detection_data)
    feature_name = cancer_detection_data.columns
    # print(feature_name)
    X = data_arr[:, :41]
    y_label = data_arr[:, 41]

    print(y_label)
    lb = LabelEncoder()
    y_label = lb.fit_transform(y_label)
    print(y_label)

    # estimator = SVC(kernel="linear")
    estimator = ExtraTreesClassifier(n_estimators=300)
    selector = RFE(estimator, n_features_to_select=1, verbose=1)
    selector = selector.fit(X, y_label)

    print(feature_name[selector.ranking_ - 1])
    print(selector.ranking_)

    p = matplotlib.rcParams
    p["figure.subplot.bottom"] = 0.2

    plt.figure()
    plt.title("Feature importances")

    res = np.argsort(selector.ranking_)

    plt.bar(range(41), 42 - selector.ranking_[res],
            color="r", align="center")

    plt.xticks(range(41), feature_name[selector.ranking_[res] - 1], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


frank_on_data2()