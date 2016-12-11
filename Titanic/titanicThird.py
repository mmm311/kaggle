'''
Created on 2016年12月7日

@author: liu
'''
import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../TitanicData/train.csv'))
header = csv_file_object.__next__()
data = []

for row in csv_file_object:
    data.append(row)
data = np.array(data)

import pandas as pd
df = pd.read_csv('../TitanicData/train.csv',header = 0)

# import pylab as P
# df['Age'].hist(bins = 16)
# df['Age'].dropna().hist(bins = 16, range=(0,80), alpha = 0.5)
# P.show()
# df['Gender'] = df['Sex'].map({'female' : 0,'male' : 1}).astype(int)
# print(df.head())
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) &\
                              (df['Pclass'] == j + 1)]['Age'].dropna().median()
df['AgeFill'] = df['Age']

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass== j +1),\
               'AgeFill']= median_ages[i,j]

print(df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head())

