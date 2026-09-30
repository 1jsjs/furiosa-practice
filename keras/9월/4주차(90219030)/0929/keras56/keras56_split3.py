# 로스는 0.1 이하
# 결과는 [101, 102, 103, 104, 105, 106]의 근사치가 나오면 됨
# 106.0x ~ 105.9x 까지 인정
"""
a = np.array(1,101)
x_predict = np.array(94,106)
"""
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

a = np.arange(1, 101, dtype=np.float32) / 100
size = 6

def split(dataset, size):
    xs, ys = [], []
    for i in range(len(dataset) - 2 * size + 1):
        xs.append(dataset[i:i + size])
        ys.append(dataset[i + size:i + 2 * size])
    return np.array(xs), np.array(ys)

x, y = split(a, size)
x = x.reshape(-1, size, 1)

print(x.shape, y.shape)  # (89, 6, 1), (89, 6)

model = Sequential()
model.add(LSTM(units=15, input_shape=(size, 1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(6))  # 다음 6개 값을 한 번에 출력

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, verbose=1)

loss = model.evaluate(x, y, verbose=1)
print('loss:', loss)  # 100으로 나눈 데이터 기준

# 입력은 95~100 여섯 개
x_predict = (np.arange(95, 101, dtype=np.float32) / 100)
x_predict = x_predict.reshape(1, size, 1)

y_predict = model.predict(x_predict, verbose=0)[0] * 100
print('결과:', y_predict)