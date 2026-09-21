# 40-1 copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print (x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print (x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
# print (np.max(x_train), np.min(x_train)) #255 0
# print (np.max(x_test), np.min(x_test)) #255 0
# print (np.max(y_train), np.min(y_train)) #9 0
# print (np.max(y_test), np.min(y_test)) #9 0

########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
# print (y_train.shape, y_test.shape) # (60000,) (10000,)
# # print (np.max(x_test), np.min(x_test)) #1.0 -1.0
# print (np.max(x_test), np.min(x_test)) #1.0 -1.0

# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# print (x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

#OneHot
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1) #1. 내용 (값) 2. 순서 바뀌지 않는다.
y_train = ohe.fit_transform(y_train) 
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)


"""
함수형 모델을 구성할 때에는 shape 꼭 잡아주기
Input, Output 명령어 기억하기
"""
######## 2. 모델구성
inputs = Input(shape =(28,28,1))

x = Conv2D(64, (3,3), activation='relu') (inputs)
x = MaxPool2D()(x)
x = Conv2D(filters=32, kernel_size=(3,3),  activation='relu') (x)
x = Dropout(0.2)(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu') (x)
x = MaxPool2D()(x)
x = Conv2D(filters=16, kernel_size=(2,2), activation='relu') (x)
x = Dropout(0.2)(x)
x = Conv2D(filters=16, kernel_size=(2,2), activation='relu') (x)
x = Dropout(0.2)(x)
x = Conv2D(filters=16, kernel_size=(2,2), activation='relu') (x)
x = GlobalAveragePooling2D()(x)
x = Dense(units=32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=16, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

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

"""
0917- GlobalAveragePooling(각 채널 전체의 평균을 내서 Flatten처럼 1차원으로 변환)
loss : 0.026793481782078743
acc : 0.9933000206947327
accuracy score : 0.9933
걸린 시간 : 279.27 s
"""