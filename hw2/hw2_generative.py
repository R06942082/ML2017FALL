import csv
import numpy as np


#84.051% 19700 1.43
#84.104% 15000 1.405
#84.07% 27925 

trainNumber=279250
datalist0 = []
datalist1 = []
ans = []
row = csv.reader(open('X_train.csv', 'r', encoding='big5'), delimiter=",")
rowans = csv.reader(open('Y_train.csv', 'r', encoding='big5'), delimiter=",")

# 取Y
i = -1  # 第一排沒資料
for a in rowans:
    if i == -1:
        i = 0
        continue
    ans.append(int(a[0]))

# 取X
i = -1  # 第一排沒資料
for r in row:
    if i == -1:
        i = 0
        continue
    if(ans[i] == 0):  # 答案是0(<50K)
        datalist0.append([])
        for j in range(106):
            datalist0[len(datalist0)-1].append(float(r[j]))
    else:  # 答案是1(>=50K)
        datalist1.append([])
        for j in range(106):
            datalist1[len(datalist1)-1].append(float(r[j]))
    i += 1
    if i==trainNumber:
    	break
data0 = np.array(datalist0).transpose()
data1 = np.array(datalist1).transpose()
max0=np.amax(data0, axis=1)
max1=np.amax(data1, axis=1)
min0=np.amin(data0, axis=1)
min1=np.amin(data1, axis=1)

for i in range(6):
    if i!=2:
        data0[i]=(data0[i]-np.outer(min0[i],np.ones(data0.shape[1])))/(np.outer(max0[i]-min0[i],np.ones(data0.shape[1])))
        data1[i]=(data1[i]-np.outer(min1[i],np.ones(data1.shape[1])))/(np.outer(max1[i]-min1[i],np.ones(data1.shape[1])))
print(data0)

length0 = data0.shape[1]  # data0的數量
length1 = data1.shape[1]  # data1的數量
# 找平均數
ulist0 = []
ulist1 = []
for i in range(106):
    ulist0.append(np.sum(data0[i])/length0)
    ulist1.append(np.sum(data1[i])/length1)

u0 = np.array(ulist0)
u1 = np.array(ulist1)
sigma0 = np.dot(data0-np.outer(u0, np.ones(length0)), (data0-np.outer(u0, np.ones(length0))).transpose())/length0
sigma1 = np.dot((data1-np.outer(u1, np.ones(length1))), (data1-np.outer(u1, np.ones(length1))).transpose())/length1
sigma = (sigma0*length0+sigma1*length1)/(length0+length1)  # 共用sigma，比例分配

w = np.dot((u0-u1).transpose(), np.linalg.pinv(sigma)).transpose()
b = -0.5*np.dot(np.dot(u0, np.linalg.pinv(sigma)), u0)+0.5*np.dot(np.dot(u1, np.linalg.pinv(sigma)), u1)+np.log(length0/length1)
print("w="+str(w))
print("b="+str(b))

test = []
rowtest = csv.reader(open('X_test.csv', 'r', encoding='big5'), delimiter=",")
i = -1  # 第一排沒資料
for r in rowtest:
    if i == -1:
        i = 0
        continue
    test.append([])
    for j in range(106):
        test[len(test)-1].append(float(r[j]))
test = np.array(test).transpose()

test_max=np.amax(test,axis=1)
test_min=np.amin(test,axis=1)
for i in range(6):
    if i!=2:
        test[i]=(test[i]-np.outer(test_min[i],np.ones(test.shape[1])))/(np.outer(test_max[i]-test_min[i],np.ones(test.shape[1])))



z = np.dot(w.reshape(1, 106), test)+b
z=z.reshape(z.shape[1])
print( 1/(1+np.exp(-z[4])))
pre = []
for i in range(len(z)):  # 從1開始跑
    pre.append([str(i+1)])
    if 1/(1+np.exp(-z[i])) > 0.5:
        pre[i].append(0)
    else:
        pre[i].append(1)
filename = "predict.csv"
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(z)):
    s.writerow(pre[i])
text.close()
