# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:57:38 2023

@author: premchand
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
dataset = np.cos(np.arange(1000)*(20*np.pi/1000))
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
look_back = 20

dataset = (dataset+1) / 2.

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]
 
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX.shape)
print(trainY.shape)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
'''
    trainX.shape =  (780, 20, 1)
    testX.shape =  (180, 20, 1)
    trainY.shape =  (780,)
    testY.shape =  (180,)
'''

batch_size = 1

model = Sequential()
model.add(LSTM(32, input_shape=(20, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
 
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(trainX,trainY,batch_size = batch_size,epochs=30, verbose=2)

x = np.vstack((trainX[-1][1:],(trainY[-1])))#vstack就是竖着拼起来
preds=[]
pred_num = 500

for i in np.arange(pred_num):
    pred = model.predict(x.reshape((1,-1,1)),batch_size = batch_size)
    preds.append(pred.squeeze())
    x = np.vstack((x[1:],pred))
    
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(pred_num),np.array(preds),'r',label='predctions')
cos_y = (np.cos(np.arange(pred_num)*(20*np.pi/1000))+1)/ 2.
plt.plot(np.arange(pred_num),cos_y,label='origin')
plt.legend()
plt.show()

dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back =40
dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

dataset = (dataset+1) / 2.
 
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]
 
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
 
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

batch_size = 1
model2 = Sequential()
model2.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model2.add(Dropout(0.2))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer='adam')
for i in range(30):
    model2.fit(trainX, trainY, epochs=1, batch_size=batch_size,  shuffle=False)
    model2.reset_states()
x = np.vstack((trainX[-1][1:],(trainY[-1])))
preds = []
pred_num = 500
for i in np.arange(pred_num):
    pred = model2.predict(x.reshape((1,-1,1)),batch_size = batch_size)
    preds.append(pred.squeeze())
    x = np.vstack((x[1:],pred))

plt.figure(figsize=(12,5))
plt.plot(np.arange(pred_num),np.array(preds),'r',label='predctions')
cos_y = (np.cos(np.arange(pred_num)*(20*np.pi/1000))+1)/ 2.
plt.plot(np.arange(pred_num),cos_y,label='origin')
plt.legend()
plt.show()

dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back =40
dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

dataset = (dataset+1) / 2.
 
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]
 
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
 
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

batch_size = 1
model3 = Sequential()
model3.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model3.add(Dropout(0.3))
model3.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model3.add(Dropout(0.3))
model3.add(Dense(1))
model3.compile(loss='mean_squared_error', optimizer='adam')
for i in range(100):
    print(i)
    model3.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    model3.reset_states()
    
x = np.vstack((trainX[-1][1:],(trainY[-1])))
preds = []
pred_num = 500
for i in np.arange(pred_num):
    pred = model3.predict(x.reshape((1,-1,1)),batch_size = batch_size)
    preds.append(pred.squeeze())
    x = np.vstack((x[1:],pred))
 
# print(preds[:20])
# print(np.array(preds).shape)
plt.figure(figsize=(12,5))
plt.plot(np.arange(pred_num),np.array(preds),'r',label='predctions')
cos_y = (np.cos(np.arange(pred_num)*(20*np.pi/1000))+1)/ 2.
plt.plot(np.arange(pred_num),cos_y,label='origin')
plt.legend()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas_profiling as pp

from sklearn import metrics

# NN models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('D:/PYTHON/Inspire/heart dESIS/heart.csv') # kaggle

target_name = 'target'
data_target = data[target_name]
data = data.drop([target_name], axis=1)

train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

# get mean and std from training data
mean = np.mean(train)
std = np.std(train)

# normalization
train = (train-mean)/(std+1e-7)
test = (test-mean)/(std+1e-7)

# split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.2, random_state=0)


# As NN is sensitive to its initialization and train_test_split splits validation set randomly,
# the results may vary for each trial...
# and so suggest to run several times...
def build_ann(optimizer='adam'):

    # Initializing the ANN
    ann = Sequential()

    # Adding the input layer and the first hidden layer of the ANN
    ann.add(Dense(units=32, kernel_initializer='he_normal', activation='relu', input_shape=(len(train.columns),)))
    # Adding the output layer
    ann.add(Dense(units=1, kernel_initializer='he_normal', activation='sigmoid'))

    # Compiling the ANN
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return ann


opt = optimizers.Adam(lr=0.001)
ann = build_ann(opt)

# Training the ANN
history = ann.fit(Xtrain, Ztrain, batch_size=16, epochs=200, validation_data=(Xval, Zval))

# Predicting the Train set results
ann_prediction = ann.predict(train)
ann_prediction = (ann_prediction > 0.5)*1 # convert probabilities to binary output

# Compute error between predicted data and true response and display it in confusion matrix
acc_ann1 = round(metrics.accuracy_score(target, ann_prediction) * 100, 2)
print(acc_ann1)

# Predicting the Test set results
ann_prediction_test = ann.predict(test)
ann_prediction_test = (ann_prediction_test > 0.5)*1 # convert probabilities to binary output

# Compute error between predicted data and true response and display it in confusion matrix
acc_test_ann1 = round(metrics.accuracy_score(target_test, ann_prediction_test) * 100, 2)
print(acc_test_ann1)
