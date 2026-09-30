#57-1 copy
# flatten 적용해서 1번과 성능비교
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터 
# 55-2 copy
#원데이터 x =[1,2,3,4,5,6,7,8,9,10,11,12,20,30,40,50,60]
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1) 

# split_x 로 전처리해놓고

#2. 모델 구성
model = Sequential()
model.add (LSTM(units=10, input_shape=(3,1), return_sequences = True))
model.add(LSTM(5, return_sequences = True,))
model.add(LSTM(5, return_sequences = True))
model.add(LSTM(5, return_sequences = True))
model.add(LSTM(5, return_sequences = True))
model.add(LSTM(5, return_sequences = True))
model.add(LSTM(5, return_sequences = True))

model.add(Flatten())     
model.add(Dense(8,))
model.add(Dense(1,))

model.summary()
# exit()
#.3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100,)

#4. 예측, 평가
loss = model.evaluate(x, y)
print('7층 loss:', loss)

"""
1층 loss: 782.0977783203125
2층 loss: 786.1834716796875
3층 loss: 805.0968017578125

7층 loss: 799.7151489257812
"""