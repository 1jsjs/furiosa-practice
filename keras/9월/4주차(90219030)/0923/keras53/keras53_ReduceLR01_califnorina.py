#  52-1 copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler #scaler 종류들
from sklearn.preprocessing import RobustScaler #scaler 종류들
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import time

path_save = "./_save/keras53_ReduceLR01_cali/"

#1.데이터
datasets = fetch_california_housing ()
x = datasets.data

y = datasets.target
# datasets 을 x와 y로 분리
print (x.shape, y.shape) #(20640, 8) (20640,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit (x) #x값은 모두 민맥스 스케일러 할 준비를 해라
x = scaler.transform (x) #변환시키기
print (x)
print (np.min(x), np.max(x)) #0.0 1.0000000000000002
# exit()
x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.7,
                                                    #  shuffle=True, 
                                                     random_state=42)
print (x_train)


#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add (Dense(10, input_dim = 8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001 # default 값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = 'mse', optimizer = Adam(learning_rate=learning_rate))

es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=40,
    restore_best_weights=True
)
rlr = ReduceLROnPlateau (
    monitor = 'val_loss',
    mode='auto',
    patience=40,
    factor=0.5,
)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k53_", date, filename])
# 파일명 예
# "./_save/keras30/" + "k30_", 0914_1147, 에포수-val_loss의 소수 4번째자리까지.keras
mcp = ModelCheckpoint ( 
    monitor='val_loss',
    mode='auto',
    save_best_only = True,
    filepath =filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 1000, batch_size = 16, validation_split = 0.2, callbacks=[es, mcp, rlr], verbose=1)
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
MSE:  0.5314318531388961
r2 :  0.5950473528883446
loss : 0.5314317941665649
"""

"""
standard
MSE:  0.526411327184144
r2 :  0.5988730085453423
loss : 0.5264114737510681
걸린 시간: 15.0
"""

"""
MaxAbsScaler
MSE:  0.5404278987149306
r2 :  0.5881923394975681
loss : 0.5404278635978699
걸린 시간: 86.44

#RobustScaler 


# optimizer - learning_rate = 0.0001
MSE:  0.5280966475447295
r2 :  0.5975887894357439
loss : 0.5280964970588684
걸린 시간: 73.21


#reduceLR - learning_rate = 0.01
MSE:  0.5329200095578338
r2 :  0.5939133732866222
loss : 0.5329201221466064
걸린 시간: 233.65
"""
