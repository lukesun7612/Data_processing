#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: description1.py
@time: 2019/10/23 14:57
"""
import numpy as np
import pandas as pd
import os
import missingno as msno
from matplotlib import pyplot as plt



inputdir = 'D:/result/309'
outputdir = 'D:/result/description.csv'
usecol = ['GPS speed(km/h)', 'RPM', 'Accelerator pedal position', 'Brake times']
def fun(x):
    if x > 1:
        return 0
    else:
        return x

def diff_value(df_column_name):
    a = []
    for m, n in enumerate(df_column_name):
        if n > 0:
            a.append(n)
    if len(a) == 0:
        return 0
    else:
        return a[-1] - a[0]

def range(longitude,latitude):
    a = 10**(-6)*np.array(longitude.min(), latitude.min())
    b = 10**(-6)*np.array(longitude.max(), latitude.max())
    dist = np.linalg.norm(b - a)
    return dist
def avgBrake(dataframe):
    n = []
    if dataframe['Brake times'].sum() > 2*diff_value(dataframe['Integral kilometer']):
        n = np.mean([dataframe['Brake switch'].sum(), dataframe['Brake times'].sum()])
    else:
        n = dataframe['Brake times'].sum()
    return n

results = pd.DataFrame()
for i, file in enumerate(os.listdir(inputdir)):
    print(i, file)

    filepath = os.path.join(inputdir, file)
    df = pd.read_csv(filepath)
    df = df.loc[df['Longitude'].apply(lambda x: x > 0)].loc[df['Latitude'].apply(lambda y: y > 0)]
    df['Brake switch'] = df['Brake switch'].apply(lambda x: fun(x))


    df1 = df.loc[df['GPS speed(km/h)'].apply(lambda x: x>40)]



    result = pd.DataFrame()
    for j in usecol:
        res = pd.DataFrame(df[j].describe()).T
        ind = res._stat_axis.values.tolist()
        col = res.columns.values.tolist()
        res = pd.DataFrame(res.iloc[0].values.reshape(1, 8), index=[file[:11]],
                           columns=pd.MultiIndex.from_product([ind, col]))
        result = pd.concat([result, res], axis=1)
    result.insert(0, column='Range', value=range(df['Longitude'], df['Latitude']))
    result.insert(0, column='Brakes', value=avgBrake(df))
    result.insert(0, column='Brake>40km/h', value=avgBrake(df1))
    # result.insert(0, column='Brakes per kilo', value=n/diff_value(df['Integral kilometer']))
    result.insert(0, column='Fuel difference', value=diff_value(df['Integral fuel consumption']))
    result.insert(0, column='Kilo difference', value=diff_value(df['Integral kilometer']))
    results = pd.concat([results, result])
results = results.loc[results['Kilo difference'].apply(lambda x: x > 5)].loc[results['Brakes'].apply(lambda y: y > 18)]
print(results)
if __name__ == '__main__':
    results.to_csv(outputdir, mode='w')
    # pass
