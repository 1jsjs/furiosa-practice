import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU

a = np.array(range(1,11))
size = 5 # timestep 사이즈

# print (a.shape) #(10,)


def split (dataset, size):
    xs = []
    ys = []
    for i in range(0,len(dataset) - size):
        x = dataset[i:i+size]
        y = dataset[i+size]

        xs.append (x)
        ys.append (y)
    return np.array(xs), np.array(ys)

x,y = split (a, 4)


print (x)
print (y)

x = np.array (x)
y =np.array (y)
print (x)
print (y)

x = x.reshape(-1, 4, 1)

print (x)


#2. 모델 구성
model = Sequential()
model.add (SimpleRNN(units=10, input_shape = (4,1))) 
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile (loss='mse', optimizer='adam')
model.fit (x, y, epochs = 100)

#4. 평가, 예측
results = model.evaluate (x, y)
print ('loss : ', results)

x_predict = np.array([7,8,9,10]).reshape (-1,4,1)
y_predict = model.predict (x_predict)

print ("[7,8,9,10]의 결과 :", y_predict)