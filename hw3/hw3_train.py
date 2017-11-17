import os
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout,Activation,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop,Adagrad,Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

train_number=28709

anst=[]
ansv=[]
train=[]
validation=[]

print("data processing...")
data=pd.read_csv(sys.argv[1])
data=data.values.transpose()
ans=np.zeros((28709,7))

for i in range(28709):
	ans[i][data[0][i]]=1
	if(i<train_number):
		train.append(np.uint8(data[1][i].split(' ')).reshape(48,48,1))
		anst.append(ans[i])
	else:
		validation.append(np.uint8(data[1][i].split()).reshape(48,48,1))	
		ansv.append(ans[i])	

train=np.array(train,dtype=np.float)
anst=np.array(anst)
validation=np.array(validation,dtype=np.float)
ansv=np.array(ansv)


print("normalizing")
dele=0
for i in range(train_number):
	max=np.max(train[i-dele])*1.0
	min=np.min(train[i-dele])*1.0
	mean=np.sum(train[i-dele])/(48*48)
	train[i-dele]=(train[i-dele]-mean)/(max-min)
	#train[i-dele]=train[i-dele]/255
	if max-min==0:
		print("Erase error picture "+str(i))
		train=np.delete(train,i-dele,0)
		anst=np.delete(anst,i-dele,0)
		dele+=1
		i+=1

try:
	print("process finished")
	for i in range(1):
		f1=200
		f2=50
		f3=3
		Dens=1000
		try:
			model=Sequential()
			#Convolution layer 1
			model.add(Conv2D(filters=f1,kernel_size=(5,5),input_shape=(48,48,1),padding='valid')) #225*44*44
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(Conv2D(filters=f1,kernel_size=(5,5),input_shape=(48,48,1),padding='valid')) #225*44*44
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2))) #225*22*22
			model.add(Dropout(0.6))

			#Convolution layer 2
			model.add(Conv2D(filters=f2,kernel_size=(5,5),padding='valid'))  #9*18*18
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(Conv2D(filters=f2,kernel_size=(5,5),padding='valid'))  #9*18*18
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2)))  #9*9*9
			model.add(Dropout(0.6))
			'''
			#Convolution layer 3
			model.add(Conv2D(filters=f3,kernel_size=(3,3),padding='valid'))  #9*5*5
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2)))  #5*5*5
			'''
			#Neuron network
			model.add(Flatten())
			model.add(Dense(Dens))
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(Dense(Dens))
			model.add(BatchNormalization())
			model.add(Activation('relu'))  
			model.add(Dropout(0.6))
			#output layer
			model.add(Dense(7,activation='softmax'))
			#compile
			model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-8),metrics=['accuracy'])

			filepath=sys.argv[2]+"\-{epoch:02d}-{val_acc:.2f}.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)
			os.system("mkdir "+"["+str(f1)+"_"+str(f2)+"_"+str(f3)+"_"+str(Dens)+"]")
			print("start trainning")
			model.fit(train, anst, epochs=200, batch_size=80,validation_split=0.1,callbacks=[checkpoint],verbose=1)
		except Exception as e:
			f = open("["+str(f1)+"_"+str(f2)+"_"+str(f3)+"_"+str(Dens)+"]Exception.txt",'w')
			f.write(str(e))
			f.close()
except Exception as e:
	f = open("Out_Exception.txt",'w')
	f.write(str(e))
	f.close()
#finally:
	#os.system("shutdown -s -t 30")




'''
model.save("hw3model.h5")
print("model saved")

loss, accuracy = model.evaluate(validation, ansv)
print('test loss: ', loss)
print('test accuracy: ', accuracy)'''
