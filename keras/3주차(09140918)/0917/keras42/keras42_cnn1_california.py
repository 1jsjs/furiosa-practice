#  keras19_overfit1_california.py copy
# import ssl
# sll._create_default_https_context = ssl._create_unverified_context
"""
MinMax Scaler : 원 값(x) - MIN / MAX - MIN
스케일러 중 가장 많이 쓰인다.
    ㄴ x의 최댓값으로 x들을 다 나눈다.
    x           MinMax X        y
    100            0.01         3
    1000            0.1         4
    10000             1         5
    0                 0         6
    -1             
     ㄴ 음수라면? 

"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#1.데이터
datasets = fetch_california_housing ()
x = datasets.data
y = datasets.target

# datasets 을 x와 y로 분리
print (x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.7,
                                                    #  shuffle=True, 
                                                     random_state=42)
# 원 데이터 전체를 스케일러 하면 안 된다. train만 scaler한다.

scaler = MinMaxScaler()
scaler.fit (x_train) #x값은 모두 민맥스 스케일러 할 준비를 해라
x_train = scaler.transform (x_train) #변환시키기
x_test = scaler.transform (x_test) #변환시키기

# print (x)
# print (np.min(x_train), np.max(x_train)) #0.0 1.0000000000000004
# print (np.min(x_test), np.max(x_test)) #-0.0010638297872338498 1.333173652694611
# exit()

x_train = x_train.reshape (-1,8,1,1)
x_test = x_test.reshape (-1,8,1,1)

print (x_train.shape)
print (x_test.shape)


#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(8,1,1), padding='same')) 
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu',)) 
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(1)) 

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2)
end_time = time.time()

#4.평가, 예측 (evaluate 는 test로 / predict 까지는 보류 / 판단은 evaluate의 loss 값)
y_predict = model.predict (x_test)
mse = mean_squared_error(y_test, y_predict)
print ("MSE: ", mse)

r2 = r2_score(y_test, y_predict)
print ("r2 : ", r2)

loss = model.evaluate (x_test, y_test)
print ('loss :', loss) #loss : 0.6562087535858154
print ("걸린 시간:", round(end_time-start_time, 2))
"""
minmax scaler
MSE:  0.3070518279475443
r2 :  0.7660255970856671
loss : 0.3070518374443054
걸린 시간: 664.39

0917 dnn -> cnn /epochs = 100
MSE:  0.30873052160083064
r2 :  0.7647464275466704
loss : 0.3087305426597595
걸린 시간: 90.89
"""
