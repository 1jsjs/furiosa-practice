# from sklearn.datasets import fetch_california_housing 가 안되는 경우
# import ssl
# sll._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time

#1.데이터
datasets = fetch_california_housing ()
x = datasets.data
y = datasets.target
# datasets 을 x와 y로 분리

print (x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.7, random_state=42)

#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add (Dense(10, input_dim = 8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2)
end_time = time.time()

#4.평가, 예측 (evaluate 는 test로 / predict 까지는 보류 / 판단은 evaluate의 loss 값)

y_predict = model.predict (x_test)
mse = mean_squared_error(y_test, y_predict)
print ("MSE: ", mse)

r2 = r2_score(y_test, y_predict)
print ("r2 : ", r2)

loss = model.evaluate (x_test, y_test)
print ('loss :', loss) #loss : 0.6562087535858154
print ("걸린 시간:", round(end_time-start_time, 2))
"""
MSE:  0.6647797461647238
r2 :  0.4934358631956909
loss : 0.6647798418998718
"""




print ("==============================================hist=====================================================")
print (hist) #지금 hist는 랩핑되어 있는 상태
print ("==============================================hist.history=====================================================")
print (hist.history) #지금 hist는 랩핑되어 있는 상태, epoch 횟수만큼 저장되어 있음 fit 함수는 loss와 val loss 값을 반환하고 있었다.
# ai 할 때는 리스트(두 개 이상은 리스트) 와 딕셔너리(key-value는 딕셔너리) 값을 많이 쓴다.
# 이것 가지고 시각화하면 더 이해를 잘 할 수 있게 되겠지?
print ("==============================================loss=====================================================")
print (hist.history['loss'])
print ("==============================================val_loss=====================================================")
print (hist.history['val_loss'])

print ("==============================================시각화하기=====================================================")

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:], c='red', label = 'loss') # y값만 넣으면 시간 순으로 그려줌
#plot: 선 긋기 / hist.history['loss'] 리스트 형태 데이터 / 순서가 없으면 1-10 순서대로 매칭이 된다. 명시하지 않으면 y값으로만 선을 긋는다.
plt.plot(hist.history['val_loss'][2:], c='blue', label = 'val_loss') # y값만 넣으면 시간 순으로 그려줌
plt.legend(loc='upper right')
plt.xlabel('epoch') #label을 어디에 표시할 건가
plt.title ('캘리포니아 LOSS')
plt.ylabel('loss')
plt.grid() #grid(격자표시) 추가
plt.show()

