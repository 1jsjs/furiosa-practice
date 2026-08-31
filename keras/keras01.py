import tensorflow as tf 

print (tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터 (데이터는 쌍으로 준비하기)
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential() #모델 선언
model.add(Dense(1, input_dim=1)) #단층 레이어

#3. 컴파일, 훈련
model.compile (loss='mse', optimizer='adam')
model.fit(x, y, epochs=200)

#4. 평가X 예측O
result = model.predict(np.array([4]))
print("4의 예측값 : ",result)