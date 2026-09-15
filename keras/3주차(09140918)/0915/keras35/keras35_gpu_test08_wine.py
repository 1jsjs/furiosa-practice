#acc 0.95 만들기
# acc_score :  0.9722222222222222
# 9월 10일 minmax scaler 적용함 -> acc_score :  0.9722222222222222 걸린 시간 :  7.45 s
# 9월 11일 standard scaler 적용함 -> acc : 0.97 acc_score :  0.9722222222222222 걸린 시간 :  23.02 s
"""
9월 11일 MaxAbsScaler
acc : 0.94
acc_score :  0.9444444444444444
걸린 시간 :  9.08 s
"""
import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

#1. 데이터
datesets = load_wine()
# print(datesets)
# # x값이 data, y값이 array
# print (datesets.DESCR)
#Number of Instances: 178
#Number of Attributes: 13 numeric
#Class Distribution: class_0 (59), class_1 (71), class_2 (48)
x = datesets.data
y = datesets['target']
# print (x.shape, y.shape) #(178, 13) (178,)
# print (np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48]))
y = pd.get_dummies(y, dtype=int) #OnehotenCoding을 pandas로 함
# print (y)
# print (y.shape)
x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     stratify=y)
# print (x.shape, y.shape) #(178, 13) (178, 3)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()

scaler.fit (x_train) #x값은 모두 민맥스 스케일러 할 준비를 해라
x_train = scaler.transform (x_train) #변환시키기
x_test = scaler.transform (x_test) #변환시키기

print (x)
print (np.min(x_train), np.max(x_train)) #0.0 1.0000000000000002
print (np.min(x_test), np.max(x_test)) #-0.05077262693156731 1.2428256070640176


#2. 모델 구성
model = Sequential()
model.add (Dense(100, input_dim=13, activation='relu'))
model.add (Dense(150,activation='relu'))
model.add (Dense(150,activation='relu'))
model.add (Dense(150,activation='relu'))
model.add (Dense(150,activation='relu'))
model.add (Dense(100,activation='relu'))
model.add (Dense(50,activation='relu'))
model.add (Dense(3,activation='softmax'))

#3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs = 2000, batch_size=3,
           validation_split=0.2,
           callbacks=[es]
           )
end_time = time.time()

#4. 평가, 예측
results = model.evaluate (x_test, y_test)
print ('loss :', results[0]) # 값이 두개 나온다. loss와 metrics의 acc // 리스트 형태로 나온다. 0번째가 loss, 1번째가 acc
print ('acc :', round(results[1], 2)) # 값이 두개 나온다. loss와 metrics의 acc // 리스트 형태로 나온다. 0번째가 loss, 1번째가 acc // acc는 값이 길게 나오기 때문에 반올림 ㄱㄱ

y_pred = model.predict (x_test)
# print (y_pred)
y_pred = np.argmax (y_pred, axis=1)
print (y_pred)
y_test = np.argmax (y_test, axis=1)
print (y_test)

accuracy_score = accuracy_score(y_test, y_pred)
print ('acc_score : ', accuracy_score)
print ('걸린 시간 : ', round(end_time - start_time, 2), 's')