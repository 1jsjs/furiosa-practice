# 실습 acc 0.67이상
"""
loss : 1.0504553318023682
acc : 0.6829000115394592
accuracy score : 0.6829
걸린 시간 : 511.9 s

1. model.add(Dense(units=8, activation='relu'))
2. model.add(Dropout(0.2))
1,2, 추가

"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print (x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print (x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
# print (np.max(x_train), np.min(x_train)) #255 0
# print (np.max(x_test), np.min(x_test)) #255 0
# print (np.max(y_train), np.min(y_train)) #9 0
# print (np.max(y_test), np.min(y_test)) #9 0

######## 1. 데이터
########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
# print (x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print (x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(x_train.min(), x_train.max()) #-1.0 1.0

# OneHot
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train) 
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)
# print (y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

######## 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3), activation="relu", )) # 30,30,64
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu',)) # 28,28,64
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 27,27,32
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 26,26,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 25,25,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 24,24,32
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 23,23,32
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(10, activation='softmax')) #원하는 shape상태는 (10,)이다.

# model.summary()
# 3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam',
               metrics=['acc'])
start_time = time.time()
model.fit (x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.1)
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)
y_test = np.argmax(y_test, axis=1) #.reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')