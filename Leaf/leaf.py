'''
Created on 2016年12月6日

@author: liu
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../LeafData/train.csv',index_col = 0)
testData = pd.read_csv('../LeafData/test.csv', index_col = 0)

# Extract features & labels
data = data.sample(frac= 1)
feature = data[data.columns[1:193]]
labels = data['species']

# normalize features
feature = feature / np.linalg.norm(feature, axis = 0)
testData = testData / np.linalg.norm(testData, axis = 0)


trainFeatures = feature.iloc[:891,:]
validFeatures = feature.iloc[891:,:]
trainLabels = labels.iloc[:891]
validLabels = labels.iloc[891:]

#Computes distance(s) between a query and training point(s)
def computeDistances(trainData, query):
    distances = np.sqrt(np.sum((trainData - query)**2, axis=1))
    return distances
distances = computeDistances(trainFeatures, validFeatures.iloc[1,:])
print(trainLabels.loc[distances.argmin()])