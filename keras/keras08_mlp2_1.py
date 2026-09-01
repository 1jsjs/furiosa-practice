import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])
# 원래 x 데이터는 3행 10열이 나와서 전치 해야 함
x = x.T #열과 행 전치
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델 구성
model = Sequential()

model.add(Dense(10, input_dim = 3))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 5)

#4.평가예측
loss = model.evaluate(x, y)
print ('loss = ',loss) #loss =  0.002308130729943514

results = model.predict(np.array([[10, 1.3, 0]]))
print ('[10, 1.3, 0]의 결과 : ', results) #[10, 1.3, 0]의 결과 :  [[10.085881]]