#53-12 copy
"""
0923 ReduceLR - learning_rate = 0.0001
loss : 0.29250991344451904
acc : 0.8996000289916992
accuracy score : 0.8996
걸린 시간 : 350.45 s

#sparse_categorical_crossentropy
loss : 0.2806132733821869
acc : 0.9014000296592712
accuracy score : 0.9014
걸린 시간 : 440.13 s

"""
# 실습 만들어보기 mnist 변환 시킨 데이터임 걍
# acc 0.92 이상 만들어보기

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D

from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
"""
print (x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print (x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
print (np.max(x_train), np.min(x_train)) #255 0
print (np.max(x_test), np.min(x_test)) #255 0
print (np.max(y_train), np.min(y_train)) #9 0
print (np.max(y_test), np.min(y_test)) #9 0
"""

######## 1. 데이터
########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
# print (y_train.shape, y_test.shape) # (60000,) (10000,)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print (x_train.shape, x_test.shape)


######## 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(28,28,1),)) # 26=(28-3+1),26=(28-3+1),64 가 됨
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu',)) #24=(26-3+1),24=(26-3+1),32
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 23=(24-2+1),23=(24-2+1),32
model.add(MaxPool2D())
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 22,22,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 21,21,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(10, activation='softmax')) #원하는 shape상태는 (10,)이다.

# 3. 컴파일, 훈련
learning_rate = 0.0001

model.compile (loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
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

path_save= "./_save/keras63_ReduceLR12_fashion/"
filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k53_", date, filename])
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
model.fit (x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.1, callbacks=[es,mcp,rlr])
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')
