from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1.데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1)) #단층 레이어

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 100)

#4.평가O 예측O
loss = model.evaluate(x, y)
print ("loss : ", loss)

result = model.predict(np.array([1,2,3,4,5,6,7]))
print ("7의 예측값 : ", result)

#26.09.01
# 7의 예측값 :  [[ -1.4305972]
#  [ -2.960354 ]
#  [ -4.4901114]
#  [ -6.0198684]
#  [ -7.5496254]
#  [ -9.079382 ]
#  [-10.609139 ]]