#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: correlation.py
@time: 2020/2/24 20:36
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy import stats



input = 'D:/result/data1.csv'
df = pd.read_csv(input, header= None,  skiprows=[0])
df = df.drop([62],1)
df.columns = ['ID', 'GPS time', 'ECU time', 'Accessories status', 'Longitude', 'Latitude', 'GPS speed(km/h)',
                  '1-ECU fuel consumption(L)', '2-Integral fuel consumption(L)', 'Current fuel capacity(%)',
                  '1-Odometer kilometer(km)', '2-ECU kilometer(km)', '3-GPS kilometer(km)',
                  'Selected fuel consumption(L)', 'Selected kilometer(L)', 'Selected speed(km/h)',
                  'Service stop status', 'Odometer speed', 'Wheel speed', 'Engine torque mode',
                  'Percentage of torque on driving instructions', 'Actual percentage of engine torque', 'RPM',
                  'Coolant temperature', 'Oil pressure', 'ECU fuel consumption', 'Accelerator pedal position',
                  'Parking brake switch', 'Clutch switch', 'Brake switch', 'Urea tank level',
                  'Urea tank temperature', 'Engine input voltage', 'Ignition switch voltage',
                  'Cumulative engine running time', 'Cumulative engine revolutions', 'Engine fuel rate',
                  'Instantaneous engine fuel rate', 'Average fuel consumption', 'Particle catcher inlet pressure',
                  'Relative boost pressure', 'Intake manifold temperature', 'Absolute boost pressure',
                  'Discharge temperature', 'Atmospheric pressure', 'Cabin temperature', 'Atmospheric temperature',
                  'Cold start light', 'Kilometers of this driving cycle', 'Total kilometers', 'Fuel contains water',
                  'Target gear', 'Actual speed ratio', 'Current gear', 'Gauge fuel level',
                  'Odometer subcounts kilometer', 'Total odometer kilometer', 'Integral kilometer',
                  'Integral fuel consumption', 'Interval brake times', 'Merger marks', 'Compensation transmission']
df.set_index('ID',inplace=True)

a, b = df['Interval brake times'], df['Accelerator pedal position']
df1 = pd.concat([a,b],1)
df1 = df1.dropna()
print(a.corr(b, method='spearman'))
# sns.regplot(x='Brakes', y='Accelerator', data=df1)
# plt.show()
# x1 = stats.pearsonr(a, b)

x2 = stats.spearmanr(a, b)
#
x3 = stats.kendalltau(a, b)
#
print(x2,x3)
if __name__ == '__main__':
    pass