# 실습 만들어보기 mnist 변환 시킨 데이터임 걍
# acc 0.92 이상 만들어보기

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D

from sklearn.preprocessing import OneHotEncoder
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
print (y_train.shape, y_test.shape) # (60000,) (10000,)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print (x_train.shape, x_test.shape)

#OneHot
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train) 
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)
print (y_train.shape, y_test.shape)

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
model.compile (loss='categorical_crossentropy', optimizer='adam',
               metrics=['acc'])
start_time = time.time()
model.fit (x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.1)
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
"""
0916
loss : 0.32399019598960876
acc : 0.9115999937057495
accuracy score : 0.9116
걸린 시간 : 319.79 s

드롭아웃 추가, 배치 사이즈 256로 증가
loss : 0.25558337569236755
acc : 0.9186999797821045
accuracy score : 0.9187
걸린 시간 : 298.08 s

model.add(Dense(units=8, activation='relu')) 추가
loss : 0.2831413149833679
acc : 0.9146000146865845
accuracy score : 0.9146
걸린 시간 : 281.42 s

배치 사이즈 128으로 복구
loss : 0.3088945746421814
acc : 0.9088000059127808
accuracy score : 0.9088
걸린 시간 : 780.01 s

0917 - maxpooling 추가
loss : 0.2715640068054199
acc : 0.9059000015258789
accuracy score : 0.9059
걸린 시간 : 342.45 s

0917- GlobalAveragePooling(각 채널 전체의 평균을 내서 Flatten처럼 1차원으로 변환)
loss : 0.26512306928634644
acc : 0.9072999954223633
accuracy score : 0.9073
걸린 시간 : 534.79 s
"""