# -*- coding: utf-8 -*-
# @Time    : 4/30/18 8:09 PM
# @Author  : Jason Lin
# @File    : gen_sex_label.py
# @Software: PyCharm
import pandas as pd
import numpy as np

data = pd.read_csv("../data/Student_DifferentCancerDetection.csv")

for idx in data.index:
    if data['Sex'][idx] == "Female":
        data['Sex_label'][idx] = 0
    else:
        data['Sex_label'][idx] = 1


print(data)

data.to_csv("../data/Student_DifferentCancerDetection.csv", index=False)