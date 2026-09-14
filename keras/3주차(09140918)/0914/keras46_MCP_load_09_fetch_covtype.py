#acc 0.93 만들기
# acc_score :  0.9466192783318848
# 9월 10일  minmax scaler 적용함 -> acc_score :  0.9590199908780324 걸린 시간 :  1332.49 s
"""
9월 11일 standard scaler 적용
loss : 0.11374743282794952
acc : 0.96
acc_score :  0.9606206380213936
걸린 시간 :  1628.23 s
"""

"""
# 26.09.11 기준 MaxAbsScaler 적용
loss : 0.11903402209281921
acc : 0.96
acc_score :  0.9562145555622488
걸린 시간 :  1211.6 s
"""

import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

path = "./_save/keras45/"

#1. 데이터
datasets = fetch_covtype()
# print (datasets.DESCR)
"""
**Data Set Characteristics:**
Classes                        7
Samples total             581012
Dimensionality                54
Features                     int
"""
x = datasets.data
y = datasets['target']
# print (x.shape, y.shape) #(581012, 54) (581012,)
# print (np.unique(y, return_counts=True)) (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
y = pd.get_dummies(y, dtype=int)
# to_categorical(y) 은 0부터 시작
# print (y.shape) #(581012, 7)
x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     stratify=y)
# print (x.shape, y.shape) #(581012, 54) (581012, 7)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler.fit (x_train) #x값은 모두 민맥스 스케일러 할 준비를 해라
x_train = scaler.transform (x_train) #변환시키기
x_test = scaler.transform (x_test) #변환시키기

print (x)
print (np.min(x_train), np.max(x_train)) #0.0 1.0
print (np.min(x_test), np.max(x_test)) #0.0 1.0050359712230217

#2. 모델 구성
model = load_model (path + "keras45k45_0914_14410093-0.114176.keras")

#3. 컴파일, 훈련

#4. 평가, 예측
results = model.evaluate (x_test, y_test)
print ('loss :', results[0]) 
print ('acc :', round(results[1], 2))

y_pred = model.predict (x_test)
print (y_pred)
y_pred = np.argmax (y_pred, axis=1)
print (y_pred)
y_test = np.argmax (y_test, axis=1)
print (y_test)

accuracy_score = accuracy_score(y_test, y_pred)
print ('acc_score : ', accuracy_score)




"""
loss : 0.1184324249625206
acc : 0.96
[[3.53482307e-07 9.99999046e-01 3.12585469e-09 ... 6.52230767e-07
  7.72465841e-11 2.67643907e-10]
 [2.55678715e-05 2.00512915e-07 4.06890112e-11 ... 4.46205323e-10
  8.86525135e-19 9.99974251e-01]
 [1.10901616e-04 1.31777711e-02 5.11784947e-07 ... 9.86710727e-01
  9.72076961e-08 6.29465946e-09]
 ...
 [1.19218420e-11 1.00000000e+00 4.25242231e-19 ... 4.02188019e-17
  1.18226573e-19 2.25747246e-18]
 [8.78550019e-03 9.91214395e-01 1.39485778e-09 ... 1.48486450e-08
  8.97022456e-09 2.18479403e-08]
 [9.99990940e-01 8.43716862e-06 1.11479277e-10 ... 1.27762831e-07
  9.10954367e-11 4.21668204e-07]]
[1 6 4 ... 1 1 0]
[1 6 4 ... 1 1 0]
acc_score :  0.9566104145331876
걸린 시간 :  1128.17 s

loss : 0.1184324249625206
acc : 0.96
[[3.53482307e-07 9.99999046e-01 3.12585469e-09 ... 6.52230767e-07
  7.72465841e-11 2.67643907e-10]
 [2.55678715e-05 2.00512915e-07 4.06890112e-11 ... 4.46205323e-10
  8.86525135e-19 9.99974251e-01]
 [1.10901616e-04 1.31777711e-02 5.11784947e-07 ... 9.86710727e-01
  9.72076961e-08 6.29465946e-09]
 ...
 [1.19218420e-11 1.00000000e+00 4.25242231e-19 ... 4.02188019e-17
  1.18226573e-19 2.25747246e-18]
 [8.78550019e-03 9.91214395e-01 1.39485778e-09 ... 1.48486450e-08
  8.97022456e-09 2.18479403e-08]
 [9.99990940e-01 8.43716862e-06 1.11479277e-10 ... 1.27762831e-07
  9.10954367e-11 4.21668204e-07]]
[1 6 4 ... 1 1 0]
[1 6 4 ... 1 1 0]
acc_score :  0.9566104145331876
"""