import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
print (x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print (x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
print (np.max(x_train), np.min(x_train)) #255 0
print (np.max(x_test), np.min(x_test)) #255 0
print (np.max(y_train), np.min(y_train)) #9 0
print (np.max(y_test), np.min(y_test)) #9 0
"""
######## 1. 데이터
########## 1-1. 스케일링 1(대상 x)
############## ㄴ 0-1 사이인 minmax가 괜찮겠다. 근데 최솟값이 0이니  maxabs도 괜찮겠다. 지금은 민맥스나 맥스앱스나 같음
"""
x_train = x_train/255.
x_test = x_test/255.
print (np.max(x_train), np.min(x_train)) #1.0 0.0
print (np.max(x_test), np.min(x_test)) #1.0 0.0
"""
########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
# print (y_train.shape, y_test.shape) # (60000,) (10000,)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print (x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

#OneHot
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1) #1. 내용 (값) 2. 순서 바뀌지 않는다.
y_train = ohe.fit_transform(y_train) 
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)
print (y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

######## 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(28,28,1),)) # 26=(28-3+1),26=(28-3+1),64 가 됨
model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu',)) #24=(26-3+1),24=(26-3+1),32
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 23=(24-2+1),23=(24-2+1),32
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 22,22,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 21,21,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 20,20,16
# (N, 20,20,16) 인데 N-데이터 개수는 상관없음
# 마지막 레이어를 가기 위해서는 2차원으로 변환해야 해서 reshape한다. -> (N, 20x20x16)
# 그러면 값과 순서가 바뀌지 않고 잘 넘겨줄 수 있음 근데 flatten() 자체만으로도 쫙 펴준다.
# model.add(Flatten()) #실무에서는 global average pooling을 더 많이 쓴다.
model.add(GlobalAveragePooling2D()) #실무에서는 global average pooling을 더 많이 쓴다.
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax')) #원하는 shape상태는 (10,)이다.

model.summary()
exit()

# 3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam',
               metrics=['acc'])
start_time = time.time()
model.fit (x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.1)
end_time = time.time()

# 4. 평가, 예측
print ('==============modelevaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)
y_test = np.argmax(y_test, axis=1) #.reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')

"""
loss : 0.044855743646621704
acc : 0.9901999831199646
accuracy score : 0.9902
걸린 시간 : 228.28 s
"""
"""
epochs=100, batch_size=128, verbose=1, validation_split=0.1
loss : 0.0581037662923336
acc : 0.9907000064849854
accuracy score : 0.9907
걸린 시간 : 358.1 s
"""