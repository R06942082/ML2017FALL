import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from gensim import corpora

# read data
print("Reading training data")
mydata = pd.read_csv('Training_label.txt', sep='\n', encoding='utf-8').values
label = []
Sentence = []
for d in mydata:
    temp = d[0].split("+++$+++")
    label.append([1-int(temp[0].strip()), int(temp[0].strip())])  # [1,0] or [0,1]
    Sentence.append(temp[1].lstrip().rstrip().split(' '))  # remove left and right spaces and split

print("Making Dictionary")
dictionary = corpora.Dictionary(Sentence)
dictionary.filter_extremes(no_below=50)
#dictionary.save('BOW_Dictonary.dict')

print("Converting")
Sentence_bow = np.zeros((len(Sentence), len(dictionary)))
for i in range(len(Sentence)):
    bow = dictionary.doc2bow(Sentence[i]) 
    for k in bow:
        Sentence_bow[i][k[0]] += k[1]

del Sentence
del mydata
del temp
del bow

print("start trainning")
model = Sequential()

model.add(Dense(256, input_dim=(len(dictionary))))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2))

model.summary()
directory = "Bow_256_128_64_32_2"
filepath = "/epoch-{epoch:02d}_acc-{val_acc:.3f}.hdf5"
os.system("mkdir "+str(directory))
checkpoint = ModelCheckpoint(str(directory)+str(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Sentence_bow, label, epochs=100, batch_size=2048, validation_split=0.1, callbacks=[checkpoint])
