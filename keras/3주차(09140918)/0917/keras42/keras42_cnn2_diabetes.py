# keras12_R2_RMSE_03_diabetes.py copy

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print (x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.80, random_state=333)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform (x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform (x_test)


x_train = x_train.reshape (-1,10,1,1)
x_test = x_test.reshape (-1,10,1,1)

print (x_train.shape)
print (x_test.shape)



#2.모델구성 (input_dim = 10, 행무시 열우선)
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(10,1,1), padding='same')) 
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu',)) 
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(1)) 


#3.컴파일, 훈련
import time
model.compile (loss = 'mse', optimizer = 'adam')
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs = 1000,
           batch_size = 3, 
           validation_split = 0.2,
           callbacks = [es])
end_time = time.time()

#4.평가, 예측
loss = model.evaluate (x_test, y_test)
print ('loss : ', loss) #loss :  2690.05908203125

y_predict = model.predict (x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score (y_test, y_predict)
print ("r2 :", r2) #r2 : 0.5057520187712794

loss = model.evaluate (x_test, y_test)
print ('loss :', loss)
print ("걸린 시간:", round(end_time-start_time, 2))


#R2 기준 0.62 이상
#r2 : 0.5133829603918585

# 26.09.10 기준 minmax scaler 적용
"""
r2 : 0.42872237043373007
loss : 3028.859619140625
걸린 시간: 7.23
"""

"""
# 26.09.11 기준 standard scaler 적용
r2 : 0.3609701747001729
loss : 3388.0751953125
걸린 시간: 10.19
"""

"""
# 26.09.11 기준 RobustScaler 적용
loss :  3312.933349609375
r2 : 0.37514282411740596
loss : 3312.933349609375
걸린 시간: 8.08

# 26.09.11 기준 MaxAbsScaler 적용
loss :  4133.71923828125
r2 : 0.22033324088500428 
loss : 4133.71923828125
걸린 시간: 13.07

0917 dnn -> cnn
loss :  2732.4658203125
r2 : 0.48462562185703384
loss : 2732.4658203125
걸린 시간: 14.79
"""