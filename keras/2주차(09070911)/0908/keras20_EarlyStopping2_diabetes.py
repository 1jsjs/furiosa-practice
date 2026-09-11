# keras12_R2_RMSE_03_diabetes.py copy
"""
0908
train_size=0.70
random_state=5
10 10 10 10 1
EarlyStopping : val_loss min 20 True
epochs = 500000000
batch 10 validation_split 0.5

r2 : 0.4887286476255802
loss : 3331.63525390625
걸린 시간: 7.57
"""

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print (x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.70, random_state=5)

#2.모델구성 (input_dim = 10, 행무시 열우선)
model = Sequential()
model.add(Dense(10, input_dim = 10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping (
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
) #클래스니 이렇게

import time
model.compile (loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit (x_train, y_train, 
                  epochs = 500000000, 
                  batch_size = 10, 
                  validation_split = 0.5,
                  callbacks=[es],
                  )
end_time = time.time()

#4.평가, 예측
loss = model.evaluate (x_test, y_test)
print ('loss : ', loss) #loss :  2690.05908203125

y_predict = model.predict (x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score (y_test, y_predict)
print ("r2 :", r2) #r2 : 0.5057520187712794

loss = model.evaluate (x_test, y_test)
print ('loss :', loss)
print ("걸린 시간:", round(end_time-start_time, 2))


#R2 기준 0.62 이상
#r2 : 0.5133829603918585

import matplotlib.pyplot as plt

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


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.title('diabetes')
plt.ylabel('loss')
plt.grid()
plt.show()
