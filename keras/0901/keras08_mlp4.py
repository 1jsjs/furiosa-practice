# x값 한 값와 y값 여러 개 있는 경우 -> 가능하다.

# #[11,0,-1]의 예측값
# 요구사항 11,0,-1

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터 
# 문제와 답을 알고 있는 과적합 문제가 있다.
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1,],
             [9,8,7,6,5,4,3,2,1,0]]).T
print (x.shape) #(10,)
print (y.shape) #(10, 3)

#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim = 1))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(3))

#3.컴파일 / 학습
model.compile(loss = 'mse', optimizer = 'adam')
model.fit (x, y, epochs = 120, batch_size = 3)

#4.평가 / 예측
loss = model.evaluate(x, y)
print ('loss = ',loss) #loss =  0.0009148367680609226

results = model.predict(np.array([[10]]))
print ('[11, 0, -1]의 결과 : ', results) #[11, 0, -1]의 결과 :  [[10.968614   -0.03443395 -0.9539119 ]]
