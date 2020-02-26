#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: contact_file.py
@time: 2019/10/1 17:20
"""
import pandas as pd
import os

inputdir = 'D:/result/309'
outputfile = 'D:/result/data1.csv'
column = ['ID', 'GPS time', 'ECU time', 'Accessories status', 'Longitude', 'Latitude', 'GPS speed(km/h)',
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

bank = pd.DataFrame([], columns=column)
bank.to_csv(outputfile, mode='w', index=False)
for i, file in enumerate(os.listdir(inputdir)):

    print(i, file)

    # 读取文件
    filePath = os.path.join(inputdir, file)
    df = pd.read_csv(filePath, header=None, skiprows=1)
    if len(df) == 0:
        print('List is null')
        continue

    df.to_csv(outputfile, mode='a', index=False, header=None)

if __name__ == '__main__':
    pass
