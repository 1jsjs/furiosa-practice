from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1)) #단층 레이어 y = wx + b 단 하나

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 100)

#4.평가O 예측O
loss = model.evaluate(x, y)
print ("loss : ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print ("6의 예측값 : ", result)

# 2026.09.01
# 6의 예측값 :  [[-0.5149182]
#  [-1.128487 ]
#  [-1.7420558]
#  [-2.3556247]
#  [-2.9691935]]