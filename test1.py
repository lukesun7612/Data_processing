#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: test1.py
@time: 2019/10/4 13:38
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
# pd.set_option('display.width', 4000, 'display.max_columns', 500, 'display.max_rows', 500)
# np.set_printoptions(threshold=np.inf)
input = 'D:/result/description.csv'
df = pd.read_csv(input, header=[0,1], index_col=0)


a = df[['Range']]
b, c, d = df['GPS speed(km/h)'], df['RPM'], df['Accelerator pedal position']
b, c, d = b[['mean']], c[['mean']], d[['mean']]
x = pd.concat([a, b, c, d], axis=1)
x.columns = ['Range', 'Speed', 'RPM', 'Accelerator pedal position']
y = df[['Brakes per kilo']].values.ravel()
# y = y.rename(columns = {'mean': 'Brake times'})
# y = y.values.reshape(-1, 1)
bins = [0, 1.1, 10]
y = pd.cut(y, bins, labels=[0, 1])
# y = KBinsDiscretizer(n_bins=2, encode='ordinal').fit_transform(y)
print(pd.value_counts(y))
l1 = []
l2 = []
l1test = []
l2test = []
x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.1, random_state=400)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
for i in np.linspace(0.01,1,1000):
    lrl1 = LogisticRegression(penalty="l1",solver="saga",C=i,max_iter=1000, multi_class='multinomial', class_weight='balanced')
    lrl2 = LogisticRegression(penalty="l2",solver="lbfgs",C=i,max_iter=1000, multi_class='multinomial', class_weight='balanced')
    lrl1 = lrl1.fit(x_train_std,y_train)
    l1.append(accuracy_score(lrl1.predict(x_train_std),y_train))
    l1test.append(accuracy_score(lrl1.predict(x_test_std),y_test))
    lrl2 = lrl2.fit(x_train_std, y_train)
    l2.append(accuracy_score(lrl2.predict(x_train_std),y_train))
    l2test.append(accuracy_score(lrl2.predict(x_test_std),y_test))
graph = [l1,l2,l1test,l2test]
color = ["red","black","blue","gray"]
label = ["L1train","L2train","L1test","L2test"]
plt.figure(figsize=(6,6))
for j in range(len(graph)):
    plt.plot(np.linspace(0.01,1,1000),graph[j],color[j],label=label[j])
plt.legend(loc=0)
# plt.show()

lg = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000, multi_class='multinomial', class_weight='balanced')
# lg = LogisticRegression(C=1, penalty='l1', solver='saga', max_iter=1000, multi_class='multinomial')
lg.fit(x_train_std, y_train)
# ln.fit(x2_train, y2_train)
# R2 = ln.score(x2_test, y2_test)
accuracy = lg.score(x_test_std, y_test)
# pre1 = lg.predict_proba(x1_test_std)
# pre2 = ln.predict(x2_test)
alpha = lg.intercept_[:]
beta = lg.coef_[:]
print(accuracy,'\n',alpha,'\n',beta)



if __name__ == '__main__':
    pass