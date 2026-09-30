#  54-1 copy
"""
단방향
loss :  0.5467121005058289
[8,9,10]의 결과 : [[11.048429]]


양방향
loss :  0.0033383723348379135
[8,9,10]의 결과 : [[10.429883]]

loss :  0.001580471987836063
[8,9,10]의 결과 : [[10.577245]]
걸린 시간 :  2.1 s
"""
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],

])

y = np.array([4,5,6,7,8,9,10])
print (x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) 

print (x.shape, y.shape) #(7, 3, 1) (7,) 
print (x.shape) ##(7, 3, 1) (7,) 


#2. 모델 구성
model = Sequential()
model.add (Bidirectional(SimpleRNN(10), input_shape = (3,1)))
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
learning_rate = 0.01

model.compile (loss='mse', optimizer=Adam(learning_rate=learning_rate))

start_time = time.time()
model.fit (x, y, epochs = 100)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate (x, y)
print ('loss : ', results)

x_predict = np.array([8,9,10]).reshape (1,3,1)
y_predict = model.predict (x_predict)

print ("[8,9,10]의 결과 :", y_predict)
print ('걸린 시간 : ', round(end_time - start_time, 2), 's')