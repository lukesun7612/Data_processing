#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: test2.py
@time: 2019/11/12 12:00
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.api as sm
import matplotlib.pyplot as plt

input = 'D:/result/description.csv'
df = pd.read_csv(input, header=[0,1], index_col=0)
# df['intercept'] = 1
# z = df[['intercept']]
a = df[['Kilo difference', 'Brakes', 'Range']]
# a = df[['Kilo difference','Range']]
b, c, d = df['GPS speed(km/h)'], df['RPM'], df['Accelerator pedal position']
b, c, d = b[['mean']], c[['mean']], d[['mean']]
# x = pd.concat([a,b,c,d],1)
x = pd.concat([a, b, c], axis=1)
# X = pd.concat([a, b, c, d], axis=1)
# x.columns = ['Distance','Range', 'Speed', 'RPM', 'Accelerator']
x.columns = ['Distance', 'Brakes', 'Range', 'Speed', 'RPM']
# X.columns = ['Distance', 'Brakes', 'Range', 'Speed', 'RPM', 'Accelerator']
pd.set_option('display.max_columns', None)
# print(X.describe())

poly = PolynomialFeatures(degree=2).fit(x)
x_ = pd.DataFrame(poly.transform(x))
x_.columns = poly.get_feature_names(x.columns)
# y = df['Brake>40km/h'].values.ravel()
y = d
y.columns = ['Accelerator']
y = y.values.ravel()


fig, axes = plt.subplots(4, 5)
for p, ax in enumerate(axes.flatten()):
    ax.plot(x_.iloc[:, p+1], y, 'x')
    ax.set_xlabel(x_.columns[p+1])             #设置y轴的标签
    # ax.set_ylabel('Brakes')               #设置y轴的标签
    ax.set_ylabel('Accelerator')
    # ax.set_title(x_.columns[p+1])              #设置该子图的标题
plt.tight_layout(w_pad=-3, h_pad=-1.5)
plt.show()



# x_ = x_.drop(['Range Speed','Range','Accelerator pedal position^2','Speed^2','Distance Range','RPM^2','RPM'],1)
# from sklearn.feature_selection import mutual_info_classif as MIC
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier as RFC
# RFC_ = RFC(n_estimators =10,random_state=0)
# X_embedded = SelectFromModel(RFC_,threshold=0.005).fit_transform(x_,y.astype(int))
# print(X_embedded.shape)
# result = MIC(x,y)
# k = result.shape[0] - sum(result <= 0)
# print(k)


# drop = ['Range Speed','Range','Accelerator^2','Speed^2','Distance Range','RPM^2','RPM']
# ac, bc = [], []
# for p in drop:
#     ac.append(p)
#     bc.append(sm.OLS(y, x_.drop(ac, 1)).fit().rsquared)
# plt.figure()
# plt.plot(np.arange(0, len(drop)), bc)
# plt.show()
# linera_model = sm.OLS(y, x_)

# x1 = x_.drop(['Range','RPM','Speed^2','RPM^2','Accelerator^2','Distance Range','Range Speed'],1)
x1 = x_.drop(['Range','Range RPM','Distance Speed','Distance^2','Distance Brakes','Speed RPM','Brakes^2','Brakes Range','Distance Range','RPM^2','Range^2','Range Speed'],1)
linera_model = sm.OLS(y, x1)
result1 = linera_model.fit()
# st, data, ss2 = summary_table(result1, alpha=0.05)
# print(st)
# residuals = y - result1.fittedvalues
print(result1.summary())

x_s = MinMaxScaler().fit_transform(result1.predict(x1).values.reshape(-1,1))
y_s = MinMaxScaler().fit_transform(y.reshape(-1,1))



# x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.3, random_state=400)
# sc = StandardScaler()
# sc.fit(x_train)
# x_train_std = sc.transform(x_train)
# x_test_std = sc.transform(x_test)

# ln = LinearRegression()
# ln.fit(x_, y)
# R2 = ln.score(x_, y)
# alpha = ln.intercept_
# beta = ln.coef_
# print(R2,'\n',alpha,'\n',beta)


bins = [-1, np.median(y), np.max(y)+1]
y_ = pd.cut(y, bins, labels=[0, 1])
print(pd.value_counts(y_))
# x2 = x_.drop(['Speed','Accelerator','RPM','Range','Distance^2','Range^2','Distance Speed','Distance RPM','Distance Accelerator','Distance Range','Range RPM','Range Speed','Range Accelerator','Speed Accelerator'],1)
x2 = x_.drop(['Brakes Speed','Distance','Distance Speed','RPM^2','RPM','Range','Range RPM','Distance RPM','Range^2','Speed^2','Brakes RPM','Brakes'],1)
logit_model = sm.Logit(y_, x2)
result2 = logit_model.fit()
print(result2.summary())


r1 = pd.DataFrame(x_s, columns=['x1'], index=df.index.values)
r2 = pd.DataFrame(result2.predict(x2).values, columns=['x2'], index=df.index.values)
sum = pd.concat([r1,r2,pd.DataFrame(y_s,columns=['y'],index=df.index.values)],1)
# s1 = sum.loc[sum['x1'].apply(lambda x: x<np.median(x_s))].loc[sum['y'].apply(lambda y:y<np.median(y_s))].count()
# print(s1)


sum.loc[(sum['x1']>=np.median(x_s))&(sum['y']>np.median(y_s)),'OLS']=1
sum.loc[(sum['x2']>=np.median(result2.predict(x2)))&(sum['y']>np.median(y_s)),'Logit']=1
sum.loc[(sum['OLS']==1)&(sum['Logit']==1),'share']=1
print(sum.count())
# sum.fillna(0,inplace=True)
# sum.to_csv('D:/result/risk result0.csv',mode='w')
# sum.to_csv('D:/result/risk result.csv',mode='w')

# lg = LogisticRegression(C=1e10,penalty='l2', solver='saga', max_iter=1000000, multi_class='multinomial',fit_intercept=False)
# lg.fit(x2, y_)
# accuracy = lg.score(x2, y_)
# alpha = lg.intercept_
# beta = lg.coef_
# pre = lg.predict(x2)
# print(accuracy,'\n',alpha,'\n',beta,'\n',pre)





plt.figure()
ax1 = plt.subplot(121)
plt.plot(y_s, x_s, 'o')
plt.xlabel('Intuitive judgment', fontsize = 15)
plt.xlim([0,1])
plt.xticks([0,np.median(y_s),1],['$Good$','$Median$','$Bad$'], fontsize = 12)
plt.ylabel('Predicted result', fontsize =15)
plt.ylim([0,1])
plt.yticks([0,np.median(x_s),1],['$Good$','$Median$','$Bad$'], fontsize = 12)
plt.title('Linear regression',fontsize=15)
ax = plt.gca()
ax.spines['right'].set_color('k')
ax.spines['top'].set_color('k')
plt.grid(True)
plt.rc('grid', linestyle="-", color='black')

ax2 = plt.subplot(122)
plt.plot(y_s, result2.predict(x2), 'o')
plt.xlabel('Intuitive judgment', fontsize = 15)
plt.xlim([0,1])
plt.xticks([0,np.median(y_s),1],['$Good$','$Median$','$Bad$'], fontsize = 12)
plt.ylabel('Predicted result', fontsize =15)
plt.ylim([0,1])
plt.yticks([0,np.median(result2.predict(x2)),1],['$Good$','$Median$','$Bad$'], fontsize = 12)
plt.title('Logistic regression',fontsize=15)
ax = plt.gca()
ax.spines['right'].set_color('k')
ax.spines['top'].set_color('k')
plt.grid(True)
plt.rc('grid', linestyle="-", color='black')

# plt.gca().patch.set_facecolor('0.8')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0.5))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0.5))
plt.tight_layout(w_pad=-5)
plt.show()



if __name__ == '__main__':
    pass
