# 27-1 copy
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
print (x_train)


#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add (Dense(10, input_dim = 8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 100,
                  batch_size = 32,
                  validation_split = 0.2,
                  )
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
GPU 쓸때 - Google Colab Tesla T4
실제 코드 출력 결과만 기록
MSE:  0.5301921306296609
r2 : 0.5959920251145829
loss : 0.5301920771598816
걸린 시간: 110.3
=============================
CPU 쓸때
MSE:  0.5393150348163402
r2 :  0.5890403450865206
loss : 0.5393150448799133
걸린 시간: 44.37
"""
