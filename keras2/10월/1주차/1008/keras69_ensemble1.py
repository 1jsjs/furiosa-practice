# 앙상블 모델 만들기
import time
import datetime
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import concatenate, Concatenate

#1. 데이터
x1_datasets = np.array ([range(100), range(301, 401)]).T # (100, 2)
x2_datasets = np.array ([range(101, 201), range(411, 511), range(150, 250)]).T # (100, 3)

y = np.array (range(3001, 3101))


#2-1. 모델
input1 = Input(shape=(2,))
dense1 =Dense(10, activation='relu', name='han1')(input1)
dense2 =Dense(20, activation='relu', name='han2')(dense1)
dense3 =Dense(30, activation='relu', name='han3')(dense2)
output1 =Dense(40, activation='relu', name='han4')(dense3)
# mode1 = Model(inputs=input1, outputs=output1)

#2-2 모델
input21 = Input(shape=(3,))
dense21 =Dense(50, activation='relu', name='han21')(input21)
dense22 =Dense(40, activation='relu', name='han22')(dense21)
dense23 =Dense(30, activation='relu', name='han23')(dense22)
dense24 =Dense(20, activation='relu', name='han24')(dense23)
output21 =Dense(10, activation='relu', name='ha25')(dense24)
# mode2 = Model(inputs=input21, outputs=output21)

#2-3. 모델 합치기
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense (10, name='mg2')(merge1)
merge3 = Dense (5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input21], outputs=last_output)

#3. 컴파일, 훈련
model.compile (loss='mse', optimizer='adam')


model.fit ([x1_datasets,x2_datasets], y, epochs=100,)

#4. 평가, 예측
result = model.evaluate ([x1_datasets, x2_datasets], y)
print ("loss :", result)

# x1_pred = np.array ([range(100, 106), range(400, 406)]).T 
# x2_pred = np.array ([range(200, 206), range(510, 516), range(249, 255)]).T

# 예측값까지 완성