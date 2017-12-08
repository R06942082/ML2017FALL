import pandas as pd
import numpy as np
import csv
import re
import sys
#from zhon import hanzi
from keras.models import Sequential, load_model
from gensim.models import Word2Vec
#from string import punctuation
print("Reading Testing Data")
mydata = pd.read_csv(sys.argv[1], sep='\n', encoding='utf-8').values
test = []
max_length = 0
for i in range(200000):
    temp = mydata[i][0].split(str(i)+",")[1]
    #temp = re.sub(r"[%s%s]+" % (punctuation, hanzi.punctuation), "", temp) #remove symbols (need import string.punctuation,zhon.hanzi)
    test.append(temp.lstrip().rstrip().split(' '))  # remove left and right spaces and split
    if len(test[len(test)-1]) > max_length:
        max_length = len(test[len(test)-1])

print("Word Embedding")
test_vector = []
bag = Word2Vec.load("MyDictionary.bin")
for i in range(len(test)):
    test_vector.append([])
    for j in range(max_length):
        if j < len(test[i]):
            try:
                b=bag[test[i][j]]
                test_vector[len(test_vector)-1].append(bag[test[i][j]])

                if np.std(test_vector[i][j])==0:
                    print(test[i][j])
                else:
                    test_vector[i][j] = np.array((test_vector[i][j]-np.mean(test_vector[i][j]))/np.std(test_vector[i][j])) # normalize, x=(x-mean(x))/std(x)
            except KeyError as e:
                test_vector[len(test_vector)-1].append(np.zeros(100))
        else:
            test_vector[len(test_vector)-1].append(np.zeros(100))

test_vector = np.array(test_vector)
test_vector = (test_vector-np.mean(test_vector))/np.std(test_vector)  # normalize, x=(x-mean(x))/std(x)

print("loading model")
model = load_model("MyModel.hdf5")

print("predicting")
result = np.argmax(model.predict(test_vector), axis=1)

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
