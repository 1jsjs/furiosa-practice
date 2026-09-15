#keras12_R2_RMSE_01_boston.py copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


import numpy as np
import time

#1.데이터
#이미 데이터셋에서 train, test가 분리되어 있음
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print (x_train.shape, x_test.shape) #(404, 13) (102, 13)
print (y_train.shape, y_test.shape) #(404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_test = scaler.transform (x_test)


#2.모델구성
model = Sequential()
model.add(Dense(100, input_dim = 13))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
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
model.fit (x_train, y_train, epochs = 100, batch_size =10, 
           validation_split = 0.2,
           )
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

"""
GPU 쓸때
loss(mse) :  30.844005584716797
r2 : 0.6294740369576164
mse :  <function mean_squared_error at 0x0000022AABB69FC0>
RMSE :  5.553738123116107
걸린 시간: 16.49

CPU 쓸때
loss(mse) :  22.228376388549805
r2 : 0.7329727463358845
mse :  <function mean_squared_error at 0x000001987F03BEC0>
RMSE :  4.714697896437677
걸린 시간: 14.69
"""
