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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

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
model = Sequential()
model.add (Dense(1000, input_dim=54, activation='relu'))
model.add (Dense(800,activation='relu'))
model.add (Dense(600,activation='relu'))
model.add (Dense(550,activation='relu'))
model.add (Dense(450,activation='relu'))
model.add (Dense(300,activation='relu'))
model.add (Dense(200,activation='relu'))
model.add (Dense(100,activation='relu'))
model.add (Dense(500,activation='relu'))
model.add (Dense(7,activation='softmax'))

#3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam',
               metrics=['acc'])
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time=time.time()
model.fit (x_train, y_train, epochs = 2000, batch_size=4096,
           validation_split=0.1,
           callbacks=[es])
end_time=time.time()

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
print ('걸린 시간 : ', round(end_time - start_time, 2), 's')