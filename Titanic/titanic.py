'''
Created on 2016年12月5日

@author: liu
'''
import csv as csv
import numpy as np

# 读取训练数据
csv_file_object = csv.reader(open('../TitanicData/train.csv','r'))
header = csv_file_object.__next__()

data = []
for row in csv_file_object:
    data.append(row)

data = np.array(data)

# 计算总的生存率
number_passengers = np.size(data[:, 0].astype(np.float))
number_survived = np.sum(data[:,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

#  男性生存率 女性生存率
women_only_stats = data[:,4] == 'female'
men_only_stats = data[:,4] == 'male'
women_onboard = data[women_only_stats, 1].astype(np.float)
men_onboard = data[men_only_stats, 1].astype(np.float)

proportion_women_survived =  \
                    np.sum(women_onboard) /   np.size(women_onboard)

proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)
print("女性生存率：%s" % proportion_women_survived)
print("男性生存率：%s" % proportion_men_survived)

####
#  读取测试数据
###
test_file = open('../TitanicData/test.csv','r')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

# 写入结果
prediction_file = open('../TitanicData/genderbasedmodel.csv','w',newline='')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(['PassengerId','Survived'])

for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0],'1'])
    else:
        prediction_file_object.writerow([row[0],'0'])
test_file.close()
prediction_file.close()

