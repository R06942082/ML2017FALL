import pandas as pd
import numpy as np
import pickle
import csv
import sys
from lib.helper import *
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Model, load_model
from keras.layers import Dropout, Activation, Dense, LSTM, Input, Dot
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

############################################
# sys.argv[1] : voice data(test.data)
# sys.argv[2] : options data(test.csv)
# sys.argv[3] : predicted result(pred.csv)
############################################

with open(sys.argv[1], "rb") as file:
    mydata = pickle.load(file)  # each vector length=39

print("test_voice padding")
test_voice = []
for data in mydata:
    test_voice.append([])
    for vector in data:
        test_voice[len(test_voice)-1].append(vector)
    while len(test_voice[len(test_voice)-1]) < 246:
        test_voice[len(test_voice)-1].append(np.zeros(39))
test_voice = np.array(test_voice)
del mydata

print("Reading "+str(sys.argv[2])+", do w2v and padding")
test_data = pd.read_csv(sys.argv[2], sep=',', encoding='utf-8', header=None).values
test_data = np.array([[s.rstrip().lstrip().split(" ") for s in options] for options in test_data])
test_data_vector = np.array([bag_and_padding(test_data[:, i]) for i in range(4)])
del test_data

###   model   ###
latent_dim = 128
lstm_dim = 256
voice_input = Input(shape=(None, 39))
voice_lstm = LSTM(lstm_dim,  go_backwards=True)(voice_input)
voice_lstm = Dropout(0.25)(voice_lstm)
voice_lstm = Dense(latent_dim)(voice_lstm)
voice_lstm = BatchNormalization()(voice_lstm)
voice_output = Activation('linear')(voice_lstm)

option_input = Input(shape=(None, 300))
option_lstm = LSTM(lstm_dim,  go_backwards=True)
#option_lstm = LSTM(128, go_backwards=True)
option_DNN_layer = option_lstm(option_input)
option_DNN_layer = Dropout(0.25)(option_DNN_layer)
option_DNN_layer = Dense(latent_dim)(option_DNN_layer)
option_DNN_layer = BatchNormalization()(option_DNN_layer)
option_DNN_layer = Activation('linear')(option_DNN_layer)

dot_layer = (Dot(1)([voice_output, option_DNN_layer]))
output = Dense(1, activation='sigmoid')(dot_layer)

model = Model([voice_input, option_input], output)
model.set_weights(load_model("lib/epoch-19_val_acc-0.644_real0.443.hdf5").get_weights())

print("predicting")
result = []
for i in range(4):
    result.append(model.predict([test_voice, test_data_vector[i]]))
result = np.array(result)

print("writting")
pre = []
for i in range(len(test_voice)):
    pre.append([str(i+1)])
    pre[i].append(str(np.argmax(result[:, i])))
filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "answer"])
for i in range(len(pre)):
    s.writerow(pre[i])
text.close()
