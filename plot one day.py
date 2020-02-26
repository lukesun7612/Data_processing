#!/usr/bin/env python
# -*- coding: gbk -*-

"""
@author: luke
@project: data description
@file: plot one day.py
@time: 2019/9/16 8:40
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


input = 'D:/result/309'
for i,file in enumerate(os.listdir(input)[13:14]):
    filepath = os.path.join(input, file)
    df = pd.read_csv(filepath, header=0)
    df = df.loc[df['Brake times'].apply(lambda y: y < 30)]
    df = df.loc[df['Longitude'].apply(lambda x: x > 0)].loc[df['Latitude'].apply(lambda y: y > 0)]
    df['GPS time'] = df['GPS time'].astype('datetime64')
    df = df.set_index('GPS time', drop=False)

    print(df['2018-07-03':'2018-07-08'])

    a = df.loc[df['Brake switch'].apply(lambda x: x > 0)].index.astype('str')
    a = pd.DataFrame(a).values.ravel()
    print(a)
    fig = plt.figure()
    plt.subplot(311)
    plt.plot(df['2018-07-04']['GPS time'],df['2018-07-04']['Integral kilometer'],'b-',label='Distance')
    plt.xlim('2018-07-04','2018-07-05')
    plt.ylabel('Distance(km)')
    plt.legend(loc=2)
    plt.subplot(312)
    plt.plot(df['2018-07-04']['GPS time'], df['2018-07-04']['Integral fuel consumption'], 'y-', label='Fuel')
    plt.xlim('2018-07-04', '2018-07-05')
    plt.ylabel('Fuel consumption(L)')
    plt.legend(loc=2)
    plt.subplot(313)
    plt.plot(df['2018-07-04']['GPS time'],df['2018-07-04']['GPS speed(km/h)'],'g-',linewidth = 1.5,label='Speed')
    plt.vlines(a, ymin=0, ymax=df['2018-07-04']['GPS speed(km/h)'].max(),colors='r',linestyles='--', label='Brakes')
    plt.xlim('2018-07-04', '2018-07-05')
    plt.ylabel('Speed(km/h)')
    plt.legend(loc=2)
    plt.xlabel('Time')

    # fig.legend()
    # fig, (sp1, sp2, sp3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    # sp1.plot(df['2018-07-03']['GPS time'],df['2018-07-03']['Integral kilometer'],linewidth=2, color='b', label = 'Kilometer')
    #
    # sp2.plot(df['2018-07-03']['GPS time'],df['2018-07-03']['GPS speed(km/h)'], linewidth=2, color='r', label='Speed')
    #
    # sp3.scatter(df['2018-07-03']['GPS time'],df['2018-07-03']['Brake times'], color='g', label='Brakes')
    #
    # fig.legend(['Kilo','Speed','Brakes'], loc='upper right')
    fig.tight_layout()


    plt.figure()
    plt.plot(df['Longitude'],df['Latitude'],'b-',linewidth=2,label='vehicle running track')
    plt.hlines([df['Latitude'].min(),df['Latitude'].max()],xmin=df['Longitude'].min(),xmax=df['Longitude'].max(),colors='r',linestyles='--')
    plt.text(df['Longitude'].mean(), df['Latitude'].min(), 'Longitude', family='serif', fontsize=18, style='oblique')

    plt.vlines([df['Longitude'].min(), df['Longitude'].max()], ymin=df['Latitude'].min(), ymax=df['Latitude'].max(),
               colors='r', linestyles='--')
    plt.text(df['Longitude'].max(), df['Latitude'].mean(), 'Latitude', family='serif', fontsize=18, style='oblique')

    plt.plot((df['Longitude'].min(), df['Longitude'].max()), (df['Latitude'].min(),df['Latitude'].max()),'r--')
    plt.text(df['Longitude'].mean(),df['Latitude'].mean(),'Range',family='serif',fontsize=18, style='oblique')
    plt.legend(loc=2,fontsize=18)
    plt.axis('off')
    plt.show()






if __name__ == '__main__':
    pass