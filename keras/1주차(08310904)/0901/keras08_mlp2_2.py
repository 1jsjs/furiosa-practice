import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
# x = np.array(range(10)) #range(n)이면 0부터 n-1까지 10이라면 [0 1 2 3 4 5 6 7 8 9]
x = np.array([range(10), range(21, 31,), range(201, 211)]).T
y = np.array(range(1,11))

#2.모델구성
model = Sequential()
model.add(Dense(20, input_dim = 3))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))


#3.컴파일 / 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 500, batch_size = 7)

#4.평가 / 예측
loss = model.evaluate(x, y)
print ('loss = ',loss) #loss =  0.0019612687174230814

results = model.predict(np.array([[10, 31, 211]]))
print ('[10, 31, 211]의 결과 : ', results) #[10, 31, 211]의 결과 :  [[10.91585]]

