#  keras28_Scaler01_california.py copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import time

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

# scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit (x_train) #x값은 모두 민맥스 스케일러 할 준비를 해라
x_train = scaler.transform (x_train) #변환시키기
x_test = scaler.transform (x_test) #변환시키기

print (x)
print (np.min(x_train), np.max(x_train)) #0.0 1.0000000000000004
print (np.min(x_test), np.max(x_test)) #-0.0010638297872338498 1.333173652694611
# exit()

#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add (Dense(10, input_dim = 8))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

model.summary()

path = "./_save/keras29"
model.save(path + 'keras29_1_save_mode.keras')

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 1000, batch_size = 32, validation_split = 0.2)
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
