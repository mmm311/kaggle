
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[ ]:

data = pd.read_csv('../LeafData/train.csv',index_col = 0)
testData = pd.read_csv('../LeafData/test.csv', index_col = 0)
data.head(6)


# In[ ]:

data = data.sample(frac = 1)
features = data[data.columns[1:193]]
labels = data['species']


# In[ ]:

features.head()


features = features / np.linalg.norm(features, axis=0)
testData = testData / np.linalg.norm(testData, axis=0)
features.head()

# In[ ]:

trainFeatures = features.iloc[:891,:]
validFeatures = features.iloc[891:,:]
trainLabels = labels.iloc[:891]
validLabels = labels.iloc[891:]


# In[ ]:

# Computes distance(s) between a query and training point(s)
def computeDistances(trainData, query):
    distances = np.sqrt(np.sum((trainData - query) **2 , axis=1))
    return distances
distances = computeDistances(trainFeatures, validFeatures.iloc[1,:])
trainLabels.loc[distances.argmin()]


# In[ ]:

validLabels.iloc[1]


# In[ ]:

# K nearest neighbors
def getNearestNeighbors(trainFeatures, trainLabels, query, k):
    distances = computeDistances(trainFeatures, query)
    lowestTen = distances.sort_values().head(k).index
    results = dict()
    for x in trainLabels[lowestTen]:
        if x not in results:
            results[x] = 1 / float(k)
        else:
            results[x] += 1 / float(k)
    return results


# In[ ]:

getNearestNeighbors(trainFeatures, trainLabels, validFeatures.iloc[1,:],10)


# In[ ]:

def getProbabilities(trainFeatures, trainLabels, testFeatures, k):
    leaves = np.unique(data.species.sort_values().values)
    probabilities= pd.DataFrame(0,index = testFeatures.index, columns = leaves)
    for index, query in zip(testFeatures.index, testFeatures.values):
        for key, value in getNearestNeighbors(trainFeatures, trainLabels, query, k).items():
            probabilities.loc[index, key] = value
    return probabilities


# In[ ]:

getProbabilities(trainFeatures, trainLabels, validFeatures, 100).head()


# In[ ]:

# Predictions 
def getPredictions(trainFeatures, trainLabels, testFeatures,k):
    testProbabilities = getProbabilities(trainFeatures, trainLabels,testFeatures, k)
    return testProbabilities.idxmax(axis = 1)


# In[ ]:

validPredictions = getPredictions(trainFeatures, trainLabels, validFeatures, 10)
validPredictions.head()


# In[ ]:

accuracy = sum(validPredictions == validLabels) / float(len(validLabels))
accuracy


# In[ ]:

# Log Loss Metric
def getLogLoss(trainFeatures, trainLabels, validFeatures, validLabels,k):
    leaves = np.unique(data.species.sort_values().values)
    validProbabilities = getProbabilities(trainFeatures, trainLabels, validFeatures, k)
    totalLoss = 0
    for index, row in zip(validProbabilities.index, validProbabilities.values):
        bools = np.zeros(99)
        bools[np.where(leaves==validLabels.loc[index])] = 1
        probs = np.zeros(len(row))
        for i, x in enumerate(row):
            probs[i] = np.log(max(min(x,1-10**-15),10**-15))
        totalLoss += sum(bools*probs)
    logLoss = totalLoss / -len(validProbabilities.values)
    return logLoss


# In[ ]:

for i in range(15):
    print('k = ' + str(i+1) + ': ' + str(getLogLoss(trainFeatures, trainLabels, validFeatures, validLabels, i+1)))


# In[ ]:

# Cross-Validation
def crossValidation(feature, labels, folds, k):
    n = len(features)
    totalLoss = 0
    for i in range(folds):
        start = int((n * i) / folds)
        end = int((n * (i + 1)) / folds)
        validFeatures = features.iloc[start:end,:]
        validLabels = labels.iloc[start:end]
        trainFeatures = features.iloc[0:start, :].append(feature.iloc[end:n, :])
        trainLabels = labels.iloc[0:start].append(labels.iloc[end:n])
        totalLoss += getLogLoss(trainFeatures, trainLabels, validFeatures, validLabels,k)
    averageLoss = totalLoss / folds
    return averageLoss
        


# In[ ]:

lossAll = np.zeros(15)
for i in range(15):
    lossAll[i] = crossValidation(features, labels, folds = 10, k = i+1)
  #  print("k =" + str(i + 1)+":"+str(lossAll[i]))


# In[ ]:

plt.plot(range(1, 16), lossAll,'r')
plt.xlabel('# Nearest Neighbors = k')
plt.ylabel('logLoss')
plt.show()






