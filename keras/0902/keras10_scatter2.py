import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15, 6,9,6,17,23,21])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=121)

print (x_train, x_test)
print (y_train, y_test)

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 100, batch_size = 10)
print ("=================================================")

#4.평가 예측
loss = model.evaluate (x_test, y_test)
print ('loss :', loss)

results = model.predict(x)

#시각화 하기
import matplotlib.pyplot as plt
plt.scatter (x, y) #데이터 점 찍기
plt.plot(x, results,color='red')
plt.show()

