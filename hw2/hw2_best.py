import numpy as np
import sys
import csv
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop,Adagrad,Adam
from keras.models import load_model

feature=106
model=load_model('best.h5')
print(sys.argv[0])
#-------------------test--------------------------
test = []
rowtest = csv.reader(open(sys.argv[1], 'r'), delimiter=",")
rowtest_special= csv.reader(open(sys.argv[2], 'r'), delimiter=",")
#get special feature from test.csv
test_special=[]
i=-1
for r in rowtest_special:
    if i == -1:
        i = 0
        continue
    test_special.append(r[4])

i = -1  # no data at first row
for r in rowtest:
    if i == -1:
        i = 0
        continue
    test.append([])
    test[i].append(float(test_special[i]))
    for j in range(feature):
        test[i].append(float(r[j]))
    i+=1

test = np.array(test).transpose()

test_max = np.amax(test, axis=1)
test_min = np.amin(test, axis=1)
#normalization

for i in range(7):
    if i != 3:
        test[i] = (test[i]-np.outer(test_min[i], np.ones(test.shape[1])))/(np.outer(test_max[i]-test_min[i], np.ones(test.shape[1])))

print("writting csv...")
result = model.predict(test.transpose())
pre = []
for i in range(len(result)):  # start from 1
    pre.append([str(i+1)])
    if result[i] > 0.5:
        pre[i].append(1)
    else:
        pre[i].append(0)
filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(pre)):
    s.writerow(pre[i])
text.close()
print("Success")









