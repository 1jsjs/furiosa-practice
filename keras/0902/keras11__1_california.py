# from sklearn.datasets import fetch_california_housing 가 안되는 경우
# import ssl
# sll._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
datasets = fetch_california_housing ()
x = datasets.data
y = datasets.target
# datasets 을 x와 y로 분리

print (x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.7, random_state=40)

#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add (Dense(16, input_dim = 8))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
model.compile(loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 100,)

#4.평가, 예측 (evaluate 는 test로 / predict 까지는 보류 / 판단은 evaluate의 loss 값)
loss = model.evaluate (x_test, y_test)
print ('loss :', loss) #loss : 0.6562087535858154