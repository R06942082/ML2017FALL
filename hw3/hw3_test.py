import os
import csv
import numpy as np
import pandas as pd
import sys
from math import log10
from keras.models import Sequential, load_model
from keras.layers import Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

test = []
print("data processing...")
data = pd.read_csv(sys.argv[1])
data = data.values.transpose()

for i in range(7178):
    test.append(np.uint8(data[1][i].split(' ')).reshape(48, 48, 1))
test = np.array(test, dtype=np.float)  

print("normalizing")
dele = 0
for i in range(7178):
    max = np.max(test[i-dele])*1.0
    min = np.min(test[i-dele])*1.0
    mean = np.sum(test[i-dele])/(48*48)
    test[i-dele] = (test[i-dele]-mean)/(max-min)
    # train[i-dele]=train[i-dele]/2 

print("load model")
# try:
model = Sequential()
model = load_model('-172-0.66.hdf5?dl=1')
model.summary()

print("predicting")
result = np.argmax(model.predict(test), axis=1)
print("writting")
pre = []

for i in range(result.shape[0]):
    pre.append([str(i)])
    pre[i].append(result[i])

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(pre)):
    s.writerow(pre[i])
text.close()

