#  52-1 copy
"""
# optimizer - learning_rate = 0.0001
MSE:  0.5280966475447295
r2 :  0.5975887894357439
loss : 0.5280964970588684
걸린 시간: 73.21


*LSTM
MSE:  0.3503600581570876
r2 :  0.7330245973121519
loss : 0.3503600060939789
걸린 시간: 1002.74
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler #scaler 종류들
from sklearn.preprocessing import RobustScaler #scaler 종류들
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import time

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

x_train = x_train.reshape (-1,8,1)
x_test = x_test.reshape (-1,8,1)
#8은 특성 8개를 시점처럼 놓은 것이고, 1은 각 시점의 입력 특성 수예요.
print (x_train)


#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add(LSTM(10, input_shape=(8, 1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
learning_rate = 0.0001
# learning_rate = 0.001 # default 값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = 'mse', optimizer = Adam(learning_rate=learning_rate))

es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 1000, batch_size = 32, validation_split = 0.2, callbacks=[es])
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
