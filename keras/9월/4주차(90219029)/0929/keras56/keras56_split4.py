# 데이터를 reshape 한 후, split_x 함수로 시계열 데이터로 변환
# (N, 10, 1) -> (N, 5, 2)
# 결과를 뽑는다.

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

a = np.array(range(1,101)) #벡터 데이터
size = 6


def split_x (dataset, size):
    aaa = []
    for i in range(len(dataset) -size +1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

# print (a.shape) #(100,)
data = a.reshape(-1, 2)
# print (data.shape) #(50, 2)
bbb = split_x(data, size)

# print (bbb.shape) #(45, 6, 2)

x = bbb[:, :-1, :]   # (45, 5, 2)
y = bbb[:, -1, 1]    # (45,)

print (x, y) #(50, 2)
print (x.shape, y.shape) #(50, 2)


#2. 모델 구성
model = Sequential()
model.add(LSTM(units=15, input_shape=(5,2)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, verbose=1)

#4. 예측, 평가
loss = model.evaluate(x, y)
print('loss:', loss)

x_predict = np.array(range(96,106)).reshape (1,5,2)
y_predict = model.predict(x_predict)[0]
print('결과:', y_predict)
