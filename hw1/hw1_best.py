import pandas as pd
import numpy as np
import sys

HourIndex = 0
DayIndex = 0
batch = np.zeros((240, 9), int)

#wd = [-0.11560696 ,-0.0082972 , -0.00187113 ,-0.03168826, -0.00433183, -0.07382838, 0.00550294 , 0.12950432]
#wa = 0.280258053385
w = [ 0.00288158 ,-0.00884522 ,-0.02495366 , 0.02975297 , 0.11842122, -0.34874028, 0.00517394 , 1.11638249]
b = 2.85287548146

data_size = 8


def PM(x):
    ans = b
    for i in range(data_size):
        ans += w[i]*x[i]
    return ans


def PMpro(x):
    ans = b
    average = 0
    for i in range(data_size):
        ans += w[i]*x[i]
        ans += w2[i]*x[i]**2
#        average+=x[i]
#    for i in range(data_size-1):
#        ans+= wd[i]*(x[i+1]-x[i])
    # ans+=wa*average/data_size
    return ans


def takeData():
    testData = np.genfromtxt(sys.argv[1], delimiter=',')
    x = np.zeros((240, 9), float)
    for i in range(240):
        for j in range(9):
            x[i][j] = testData[i*18+9][j+2]
    return x


data = np.zeros(240)
x = takeData()
for i in range(240):
    data[i] = PM(x[i])

out = {'value': data}
df = pd.DataFrame(data=out)
df = df.rename_axis('id')
for i in range(240):
    df = df.rename(index={i: 'id_'+str(i)})

df.to_csv(str(sys.argv[2]))
