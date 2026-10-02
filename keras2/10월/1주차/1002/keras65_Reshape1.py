#63-1 copy
"""
*sparse_categorical_crossentropy
loss : 0.025160836055874825
acc : 0.9919999837875366
accuracy score : 0.992
걸린 시간 : 369.57 s

# Reshape layer
loss : 0.039863746613264084
acc : 0.9890000224113464
accuracy score : 0.989
걸린 시간 : 405.91 s
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print (x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print (x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
print (np.max(x_train), np.min(x_train)) #255 0
print (np.max(x_test), np.min(x_test)) #255 0
print (np.max(y_train), np.min(y_train)) #9 0   
print (np.max(y_test), np.min(y_test)) #9 0

########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print (y_train.shape, y_test.shape) # (60000,) (10000,)
# print (np.max(x_test), np.min(x_test)) #1.0 -1.0
# print (np.max(x_test), np.min(x_test)) #1.0 -1.0

print (x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

# #OneHot
# ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(-1, 1) #1. 내용 (값) 2. 순서 바뀌지 않는다.
# y_train = ohe.fit_transform(y_train) 
# y_test = y_test.reshape(-1,1)
# y_test = ohe.fit_transform(y_test)
# print (y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

######## 2. 모델구성
model = Sequential()
model.add(Dense(280, input_shape=(28, 28))) # (N, 28 ,28) -> (N, 28, 280)
""" 
Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 28, 100)           2900      """
model.add (Reshape(target_shape=(28,28,10)))
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(28,28,10),)) 
model.add(MaxPool2D())
model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu',)) 
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu'))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax')) 

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001

model.compile (loss='sparse_categorical_crossentropy', # sparse를 붙이면 원핫까지 해준다. sparse 붙이면 one-hot 까지 해준다.
               optimizer=Adam(learning_rate=learning_rate),
               metrics=['acc'])

es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path_save= "./_save/keras65_ReduceLR11_mnist/"
filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k65_", date, filename])
mcp = ModelCheckpoint ( 
    monitor='val_loss',
    mode='auto',
    save_best_only = True,
    filepath =filepath,
    verbose=1,
)

rlr = ReduceLROnPlateau (
    monitor = 'val_loss',
    mode='auto',
    patience=10,
    factor=0.5,
)

start_time = time.time()
model.fit (x_train, y_train, epochs=3000, batch_size=128, verbose=1, validation_split=0.1, callbacks=[es,mcp,rlr])
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')