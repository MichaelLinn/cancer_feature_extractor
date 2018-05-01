# -*- coding: utf-8 -*-
# @Time    : 4/26/18 4:46 PM
# @Author  : Jason Lin
# @File    : mean_decrease_accuracy.py
# @Software: PyCharm
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt

# cancer detection dataset
def frank_on_data1():
    cancer_detection_data = pd.read_csv("data/Student_CancerDetection.csv")
    data_arr = np.array(cancer_detection_data)
    feature_name = cancer_detection_data.columns
    print(feature_name)
    X = data_arr[:, :9]
    y_label = data_arr[:, 9]
    scores = defaultdict(list)

    clf = ExtraTreesClassifier(n_estimators=250, random_state=0)

    # crossvalidation the scores on a number of different random splits of the data

    rs = ShuffleSplit(n_splits=100, test_size=.3, random_state=0)
    print(y_label)
    lb = LabelBinarizer()
    y_label = lb.fit_transform(y_label).T[0]
    print(y_label)

    n_sp = 0
    for train_index, test_index in rs.split(X):
        n_sp += 1
        print(n_sp)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_label[train_index], y_label[test_index]

        clf = clf.fit(X_train, Y_train)
        acc = accuracy_score(Y_test, clf.predict(X_test))
        print("accuracy: ", acc)
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = accuracy_score(Y_test, clf.predict(X_t))
            print("shuff_acc: ", shuff_acc)
            scores[feature_name[i]].append((acc - shuff_acc) / acc)

    print("Feature sorted by their score:")

    res = sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True)

    print(res)
    res = np.array(res)

    p = matplotlib.rcParams
    p["figure.subplot.bottom"] = 0.2

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(res)), res[:, 0],
            color="r", align="center")
    plt.xticks(range(len(res)), res[:, 1], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

# Different Cancer Detection Dataset
def frank_on_data2():
    cancer_detection_data = pd.read_csv("data/Student_DifferentCancerDetection.csv")
    data_arr = np.array(cancer_detection_data)
    feature_name = cancer_detection_data.columns
    # print(feature_name)
    X = data_arr[:, :41]
    y_label = data_arr[:, 41]
    scores = defaultdict(list)

    clf = ExtraTreesClassifier(n_estimators=250, random_state=0)

    # crossvalidation the scores on a number of different random splits of the data

    rs = ShuffleSplit(n_splits=500, test_size=.3, random_state=0)

    print(y_label)
    lb = LabelEncoder()
    y_label = lb.fit_transform(y_label)
    print(y_label)

    n_sp = 0
    for train_index, test_index in rs.split(X):
        n_sp += 1
        print(n_sp)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_label[train_index], y_label[test_index]

        clf = clf.fit(X_train, Y_train)
        acc = accuracy_score(Y_test, clf.predict(X_test))
        print("accuracy: ", acc)
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = accuracy_score(Y_test, clf.predict(X_t))
            print("shuff_acc: ", shuff_acc)
            scores[feature_name[i]].append((acc - shuff_acc) / acc)

    print("Feature sorted by their score:")

    res = sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True)

    print(res)
    res = np.array(res)

    p = matplotlib.rcParams
    p["figure.subplot.bottom"] = 0.2

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(res)), res[:, 0],
            color="r", align="center")
    plt.xticks(range(len(res)), res[:, 1], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


frank_on_data2()