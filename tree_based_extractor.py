# -*- coding: utf-8 -*-
# @Time    : 4/25/18 8:31 PM
# @Author  : Jason Lin
# @File    : tree_based_extractor.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import matplotlib



def frank_on_data1():
    cancer_detection_data = pd.read_csv("data/Student_CancerDetection.csv")

    data_arr = np.array(cancer_detection_data)

    feature_name = cancer_detection_data.columns
    print(feature_name)
    X = data_arr[:, :9]
    y_label = data_arr[:, 9]

    # Train a ExtraTrees Classifier
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest = forest.fit(X, y_label)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    p = matplotlib.rcParams
    p["figure.subplot.bottom"] = 0.18

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_name[indices], rotation=90)

    # plt.xlim([-1, X.shape[1]])
    # plt.savefig("fs_erf.png", dpi=100)

    plt.show()




# Different Cancer Detection
def frank_on_data2():
    cancer_detection_data = pd.read_csv("data/Student_DifferentCancerDetection.csv")
    print(cancer_detection_data)

    data_arr = np.array(cancer_detection_data)

    feature_name = cancer_detection_data.columns
    print(feature_name)
    X = data_arr[:, :41]
    y_label = data_arr[:, 41]

    # Train a ExtraTrees Classifier
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest = forest.fit(X, y_label)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest

    p = matplotlib.rcParams
    p["figure.subplot.bottom"] = 0.2

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_name[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])

    plt.show()

frank_on_data2()