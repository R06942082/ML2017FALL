import pandas as pd
import numpy as np
from keras.models import load_model
import csv
import sys

test_data = pd.read_csv(sys.argv[1], sep=',', encoding='utf-8').values

print("predicting")


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
filepath = "epoch-12_loss-0.029.hdf5"
model=load_model(filepath,custom_objects={'rmse': rmse})

result = model.predict([np.array(test_data[:, 1]), np.array(test_data[:, 2]), np.array(test_data[:, 1]), np.array(test_data[:, 2])])*5

print("writting")
pre = []
for i in range(result.shape[0]):
    pre.append([str(i+1)])
    pre[i].append(result[i][0])
filename = sys.argv[2]
text = open(filename, "w+")

s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["TestDataID", "Rating"])
for i in range(len(pre)):
    s.writerow(pre[i])
text.close()
