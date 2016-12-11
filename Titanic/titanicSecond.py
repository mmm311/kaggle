'''
Created on 2016年12月5日

@author: liu
'''
import csv as csv
import numpy as np

# 读取测试数据
csv_file_object = csv.reader(open('../TitanicData/train.csv','r'))
header = csv_file_object.__next__()
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater 
# or equal to ceiling
data[data[0:,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

# I know there were 1st , 2nd and 3rd classes on board
number_of_classes = 3

# But it's better practice to calculate this from the data
# directly

# Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[:,2]))

# Initialize the survival table with all zeros 
survival_table = np.zeros((2, number_of_classes,number_of_price_brackets))

for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        women_only_stats = data[
                                (data[:,4] == 'female')\
                                &(data[:,2].astype(np.float) == i+1)\
                                &(data[:, 9].astype(np.float) >= j * fare_bracket_size)
                                &(data[:,9].astype(np.float) < (j+1) * fare_bracket_size)
                                ,1]
        
        men_only_stats = data[
                                (data[:,4] != 'female')\
                                &(data[:,2].astype(np.float) == i+1)\
                                &(data[:, 9].astype(np.float) >= j * fare_bracket_size)
                                &(data[:,9].astype(np.float) < (j+1) * fare_bracket_size)
                                ,1]
        survival_table[0, i, j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
survival_table[survival_table != survival_table] = 0

survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1
print(survival_table)


# 测试数据
test_file = open('../TitanicData/test.csv','r')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()
prediction_file = open('../TitanicData/genderclassmodel.csv','w',newline='')
p =  csv.writer(prediction_file)
p.writerow(['PassengerId','Survived'])

for row in test_file_object:
    for j in range(number_of_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets - 1
            break
        if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size:
            bin_fare = j
            break
        
    if row[3] == 'female':
        p.writerow([row[0], "%d" % int(survival_table[0,float(row[1]) -1 , bin_fare])])
    else:
        p.writerow([row[0], "%d" % int(survival_table[1,float(row[1]) -1 , bin_fare])])
# Close out the files 
test_file.close()   
prediction_file.close()