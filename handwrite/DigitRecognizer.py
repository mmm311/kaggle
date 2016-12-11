'''
Created on 2016年10月24日

@author: liu
'''
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt, dtype
import numpy as np

def main():
    #创建训练集
    dataset = np.loadtxt(r'E:\graduate program\code\python\data\train.csv', 
                          dtype ='f8', 
                          delimiter = ',')
    target = [x[0] for x in dataset] # 第一列为label
    train = [x[1:] for x in dataset]
    
    #创建测试集
    test = np.loadtxt(r'E:\graduate program\code\python\data\test.csv',
                       dtype = 'f8',
                       delimiter = ',')
    
    #创建并且训练一个随机森林模型
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_result = [[index + 1, x] for index , x in enumerate(rf.predict(test))]
    
    
    #利用随机森林对森林对测试集进行预测，并将结果保存到输出文件中
    savetxt('result.csv',predicted_result,delimiter = ',', fmt = '%d, %d',
            header = 'ImagedId,Label',comments = '')
    
if __name__ == '__main__':
    main()
    