import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import csv
from keras.layers import Input, Dense, Merge
from keras.models import Sequential, Model, load_model
from sklearn import cluster, datasets, metrics
import sklearn
import sys


print("reading model")
model_name = "epoch-738_acc-0.010.hdf5"
encoder_inputs = Input(shape=(784,))
encoder_dense = Dense(1024, activation='relu')(encoder_inputs)
encoder_dense = Dense(512, activation='relu')(encoder_dense)
encoder_dense = Dense(256, activation='relu')(encoder_dense)
encoder_dense = Dense(128, activation='relu')(encoder_dense)
encoder_dense = Dense(32, activation='relu')(encoder_dense)
encode_model = Model(encoder_inputs, encoder_dense)
encode_model.set_weights(load_model(model_name).get_weights())
encode_model.summary()

print("making encoded_database")
database = np.load(sys.argv[1])/255
encoded_database = encode_model.predict(database)

del database
del encode_model

print("reading test_data")
test_index = pd.read_csv(sys.argv[2], sep=',').values
test_data1 = np.array([encoded_database[index[1]] for index in test_index])
test_data2 = np.array([encoded_database[index[2]] for index in test_index])

def Kmeans_clusting():
    print("Kmeans fitting encoded_database")
    kmeans = cluster.KMeans(n_clusters=2).fit(encoded_database)
    # silhouette_avg = metrics.silhouette_score(encoded_database[0:10000], cluster_labels[0:10000])
    print("Kmeans predicting test_data")
    result1 = kmeans.predict(test_data1)
    result2 = kmeans.predict(test_data2)
    print("writting")
    pre = []
    for i in range(len(result1)):
        pre.append([str(i)])
        if result1[i] == result2[i]:
            pre[i].append("1")
        else:
            pre[i].append("0")
    filename = sys.argv[3]
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["ID", "Ans"])
    for i in range(len(pre)):
        s.writerow(pre[i])
    text.close()

Kmeans_clusting()
