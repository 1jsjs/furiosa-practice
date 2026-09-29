import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU

a = np.array ([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0],
              ]).T
size = 5 # timestep 사이즈


# print (a.shape) #(10, 2)

"""
(10,2) 의 원 데이터 형태는 [[1,9],[2,8],[3,7],[4,6] ...]
(B,T,F) 
"""

def split (dataset, size):
    xs = []
    ys = []
    for i in range(0,len(dataset) - size):
        # x = dataset [:, :-1, :]
        # y = dataset [:, -1, -1] = [:, -1, 1]
        x = dataset[i:i+size]
        y = dataset[i+size]

        xs.append (x)
        ys.append (y)
    return np.array(xs), np.array(ys)
x,y = split (a, 4)

x = np.array (x)
y =np.array (y)

# print (x)
# print (y)

#2. 모델 구성
model = Sequential()
model.add (SimpleRNN(units=10, input_shape = (4,2))) 
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile (loss='mse', optimizer='adam')
model.fit (x, y, epochs = 100)

#4. 평가, 예측
results = model.evaluate (x, y)
print ('loss : ', results)

x_predict = np.array([[4,6],[5,5],[6,4],[7,3]]).reshape (-1,4,2)
y_predict = model.predict (x_predict)

print ("ypredict의 결과 :", y_predict)
