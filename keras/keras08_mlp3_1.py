# x값 여러 개와 y값 여러 개 있는 경우
# #[10,31,211]의 예측값
# 요구사항 11.00, 0.00

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10),range(21, 31), range(201, 211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1]]).T

print (x.shape)
print (y.shape)


#2.모델구성
model = Sequential ()
model.add(Dense(150, input_dim = 3))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(2))


#3.컴파일 / 학습
model.compile(loss = 'mse', optimizer = 'adam')
model.fit (x, y, epochs = 480, batch_size = 10)


#4.평가 / 예측
loss = model.evaluate(x, y)
print ('loss = ',loss) #loss =  1.2076208921740772e-09

results = model.predict(np.array([[10, 31, 211]]))
print ('[10, 31, 211]의 결과 : ', results) # [10, 31, 211]의 결과 :  [[1.10000286e+01 1.01299025e-04]]