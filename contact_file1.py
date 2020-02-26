#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: contact_file1.py
@time: 2019/10/21 10:35
"""

import pandas as pd
import os


def append_file(filedir):
    files = os.listdir(filedir)
    i = 0
    while i < len(files):
        print(files[i][:11])
        file1 = pd.read_csv(filedir + '/' + files[i], header=None, skiprows=[0], low_memory=False)
        file2 = pd.read_csv(filedir + '/' + files[i + 1], header=None, skiprows=[0], low_memory=False)
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

        # build a new column
        bank = []
        for k, v in enumerate(df['Interval brake times']):
            if v > 30:
                bank.append(30)
            else:
                bank.append(v)
        bank = pd.Series(bank, name='Brake times')
        df = pd.concat([df, bank], axis=1)
        df.to_csv('D:/result/309/' + files[i][:11] + '.csv', mode='w', index=False)
        i += 2


if __name__ == '__main__':
    filedir = 'D:/93'
    # filedir = 'D:/229'
    append_file(filedir)
