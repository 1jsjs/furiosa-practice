from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2.모델구성
model = Sequential()

#딥러닝 모델 구현
model.add(Dense(3, input_dim = 1)) #인풋은 알려줘야 함

model.add(Dense(5, input_dim = 3))

model.add(Dense(4, input_dim = 5))

model.add(Dense(3, input_dim = 4))

model.add(Dense(1, input_dim = 3))
#하이퍼파라미터튜닝하는 것



#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 100)

#4.평가O 예측O
loss = model.evaluate(x, y)
print ("loss : ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print ("6의 예측값 : ", result)


# 2026.09.01
# 6의 예측값 :  [[1.1223971]
#  [2.05571  ]
#  [2.989023 ]
#  [3.9223354]
#  [4.8556495]]