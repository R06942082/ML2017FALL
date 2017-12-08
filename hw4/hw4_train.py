from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import os
import sys
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

threshold = 44

# read data (data format:"1 +++$+++ Captain Teemo on duty!)
print("Reading training data")
mydata = pd.read_csv(sys.argv[1], sep='\n', encoding='utf-8').values
label = []
Sentence = []
max_length = 0
for d in mydata:
    temp = d[0].split("+++$+++")
    if temp[0].strip() == '0':
        label.append([1, 0])
    else:
        label.append([0, 1])
    Sentence.append(temp[1].lstrip().rstrip().split(' '))  # remove left and right spaces and split
    if len(Sentence[len(Sentence)-1]) > max_length:
        max_length = len(Sentence[len(Sentence)-1])
label = np.array(label)

print("Reading no label data")
mydata_no_label = pd.read_csv(sys.argv[2], sep='\n', encoding='utf-8').values
Sentence_no_label = []
for d in mydata_no_label:
    temp = d[0].lstrip().rstrip().split(' ')
    if len(temp) <= threshold:
        Sentence_no_label.append(temp)

print("Making Dictonary")
Sentence_combine = list(Sentence)
Sentence_combine.extend(Sentence_no_label)  # combined with no label data
Sentence_combine = np.array(Sentence_combine)
bag = Word2Vec(Sentence_combine, min_count=5,sg=0)
bag.save("Dictonary.bin")


#Release Memory
del Sentence_combine
del Sentence_no_label
del mydata
del mydata_no_label

print("Word Embedding")
Sentence_vector = []
bag = Word2Vec.load("Dictonary_all_2.bin")
for i in range(len(Sentence)):
    Sentence_vector.append([])
    for j in range(max_length):
        if j < len(Sentence[i]):
            try:
                b=bag[Sentence[i][j]]
                Sentence_vector[len(Sentence_vector)-1].append(bag[Sentence[i][j]])
                Sentence_vector[i][j] = np.array((Sentence_vector[i][j]-np.mean(Sentence_vector[i][j]))/np.std(Sentence_vector[i][j])) # normalize, x=(x-mean(x))/std(x)
            except KeyError as e:
                Sentence_vector[len(Sentence_vector)-1].append(np.zeros(100))
        else:
            Sentence_vector[len(Sentence_vector)-1].append(np.zeros(100))
del Sentence
del bag
Sentence_vector = np.array(Sentence_vector)
Sentence_vector= np.array((Sentence_vector-np.mean(Sentence_vector))/np.std(Sentence_vector))

print("Trainning")
model = Sequential()
model.add(LSTM(256, input_shape=(max_length, 100), return_sequences=True))
model.add(LSTM(128))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.summary()

directory = "modelDir"
filepath = "/epoch-{epoch:02d}_acc-{val_acc:.3f}.hdf5"
os.system("mkdir "+str(directory))
checkpoint = ModelCheckpoint(str(directory)+str(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Sentence_vector, label, epochs=50, batch_size=512, validation_split=0.1, callbacks=[checkpoint])
