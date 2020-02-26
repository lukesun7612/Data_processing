#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: luke
@project: data description
@file: data description.py
@time: 2019/9/16 0:37
"""

import numpy as np
import pandas as pd
import os



def append_file(filedir):
    usecols = ['GPS speed(km/h)', 'RPM', 'Brake times']
    files = os.listdir(filedir)
    i = 0
    results = pd.DataFrame()
    while i < len(files):
        file1 = pd.read_csv(filedir + '/' + files[i], header=None, skiprows=[0])
        file2 = pd.read_csv(filedir + '/' + files[i + 1], header=None, skiprows=[0])
        df = file1.append(file2).reset_index(drop=True)
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

        # Filter unreasonable data and build a new column
        bank = []
        for k, v in enumerate(df['Interval brake times']):
            if v > 30:
                bank.append(30)
            else:
                bank.append(v)
        bank = pd.Series(bank, name='Brake times')
        df = pd.concat([df, bank], axis=1)
        # drop and fill in the missing rows
        # df.dropna(axis=1, how='all', inplace=True)
        # df.dropna(axis=0, thresh=30, inplace=True)
        # df.fillna(method='bfill', inplace=True)
        # df.reset_index(drop=True, inplace=True)

        result = pd.DataFrame()
        for j in usecols:
            res = pd.DataFrame(df[j].describe()).T
            ind = res._stat_axis.values.tolist()
            col = res.columns.values.tolist()
            res = pd.DataFrame(res.iloc[0].values.reshape(1, 8), index=[files[i][:11]],
                               columns=pd.MultiIndex.from_product([ind, col]))
            result = pd.concat([result, res], axis=1)
        time = pd.to_datetime(df.iloc[-1]['GPS time']) - pd.to_datetime(df.iloc[0]['GPS time'])
        # result.insert(0, column='End time', value=df.iloc[-1]['GPS time'])
        # result.insert(0, column='Start time', value=df.iloc[0]['GPS time'])
        result.insert(0, column='Time difference', value=time)

        def diff_value(columnname):
            a = []
            for m, n in enumerate(df[columnname]):
                if n > 0:
                    a.append(n)
            if len(a) == 0:
                diff = 0
            else:
                diff = a[-1] - a[0]
            return diff

        total_fuel = diff_value('Integral fuel consumption')
        result.insert(0, column='Fuel difference', value=total_fuel)
        total_kilo = diff_value('Integral kilometer')
        result.insert(0, column='KM difference', value=total_kilo)
        results = pd.concat([results, result])

        i += 2

    return results


if __name__ == '__main__':
    # filedir = 'D:/93'
    filedir = 'D:/229'
    result = append_file(filedir)
    result = result.loc[result['Brake times', 'max'].apply(lambda x: x > 0)].loc[
        result['KM difference'].apply(lambda y: y > 5)]
    # pd.set_option('display.max_columns', None)
    print(result)
    # result.to_csv('D:/result/description.csv', mode='w')
    result.to_csv('D:/result/description.csv', mode='a', header=None)
