import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
#이미 데이터셋에서 train, test가 분리되어 있음
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print (x_train.shape, x_test.shape) #(404, 13) (102, 13)
print (y_train.shape, y_test.shape) #(404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_test = scaler.transform (x_test)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform (x_test)


x_train = x_train.reshape (-1,13,1,1)
x_test = x_test.reshape (-1,13,1,1)

print (x_train.shape)
print (x_test.shape)

#2.모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(13,1,1), padding='same')) 
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu',)) 
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(1)) 

#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs = 1000, batch_size =10, 
           validation_split = 0.2,
           callbacks = [es])
end_time = time.time()

print ("=================================================")
#4.평가, 예측
loss = model.evaluate (x_test, y_test, ) #default batch_size = 32 //tensorflow의 evaluate
# evaluate 를 풀어서 쓴다면?
#  y(예측값) = w * x + b
#             w * x_test + b
#      loss = Σ(y_i(test) - ŷ_i)^2 / n
# loss 할 때 x_test, y_test 둘다 쓴다.

#y^- = w(마지막 epoch 에서 나온 가중치 값으로) * x_test + b
print ('loss(mse) : ', loss) #loss :  28.8504581451416

y_predict = model.predict (x_test) #y 예측값이라서 y_predict라고 써도 됨 (이해를 돕기 위해)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict) # 인풋과 아웃풋을 다 알아야 한다.
print ("r2 :", r2)

#사이킷런에서는 원값과 테스트값만 넣으면 된다!

mse = mean_squared_error(y_test, y_predict) #사이킷런의 mse
print("mse : ", mean_squared_error)

def RMSE (y_test, y_predict): #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print ("걸린 시간:", round(end_time-start_time, 2))

#predict 에서도 w(마지막 epoch 에서 나온 가중치 값으로 구한다.

# r2 : 0.7226553409714727

# 26.09.10 기준 minmax scaler 적용
"""
loss(mse) :  16.51691246032715
r2 : 0.8015839778993096
mse :  <function mean_squared_error at 0x000002E4E93FBF60>
RMSE :  4.064100550878531
걸린 시간: 7.94
"""

# 26.09.11 기준 standard scaler 적용
"""
loss(mse) :  13.94719123840332
r2 : 0.8324537788271986
mse :  <function mean_squared_error at 0x0000020DC17FBEC0>
RMSE :  3.7345939880224734
걸린 시간: 8.51
"""

"""
# 26.09.11 기준 MaxAbsScaler 적용
loss(mse) :  20.907684326171875
r2 : 0.7488380869494919
mse :  <function mean_squared_error at 0x0000021C56283F60>
RMSE :  4.572492126173613
걸린 시간: 5.77
"""
"""
# 26.09.11 기준 RobustScaler 적용
loss(mse) :  21.33925437927246
r2 : 0.7436536576669508
mse :  <function mean_squared_error at 0x0000027717293EC0>
RMSE :  4.619443266214789
걸린 시간: 7.02

0917 dnn to cnn
loss(mse) :  33.48245620727539
r2 : 0.5977786360384756
mse :  <function mean_squared_error at 0x0000013C582C1990>
RMSE :  5.78640253465316
걸린 시간: 11.08
"""
