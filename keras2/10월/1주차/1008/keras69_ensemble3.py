# 69-2 copy
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
x3_datasets = np.array ([range(100), range(301,401), range(77,177), range(33,133)]).transpose() #(100, 4)

y1 = np.array (range(3001, 3101))
y2 = np.array (range (13001, 13101))

#2-1. 모델
input1 = Input(shape=(2,))
dense1 =Dense(10, activation='relu', name='han1')(input1)
dense2 =Dense(20, activation='relu', name='han2')(dense1)
dense3 =Dense(30, activation='relu', name='han3')(dense2)
output1 =Dense(40, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

#2-2 모델
input21 = Input(shape=(3,))
dense21 =Dense(50, activation='relu', name='han21')(input21)
dense22 =Dense(40, activation='relu', name='han22')(dense21)
dense23 =Dense(30, activation='relu', name='han23')(dense22)
dense24 =Dense(20, activation='relu', name='han24')(dense23)
output21 =Dense(10, activation='relu', name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)

#2-3 모델
input31 = Input(shape=(4,))
dense31 =Dense(15, activation='relu', name='han31')(input31)
dense32 =Dense(25, activation='relu', name='han32')(dense31)
dense33 =Dense(50, activation='relu', name='han33')(dense32)
dense34 =Dense(25, activation='relu', name='han34')(dense33)
output31 =Dense(15, activation='relu', name='han35')(dense34)
# model3 = Model(inputs=input31, outputs=output31)

#2-4. 모델 합치기
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1,output21,output31])
merge2 = Dense (10, name='mg2')(merge1)
merge3 = Dense (5, name='mg3')(merge2)

#2-5. 분기1
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(merge3)

#2-5. 분기1
last_output2 = Dense(1, name='last2')(merge3)

model = Model(inputs=[input1, input21, input31], outputs=[last_output1, last_output2])

#3. 컴파일, 훈련
model.compile (loss='mse', optimizer='adam')


model.fit ([x1_datasets,x2_datasets, x3_datasets], [y1, y2], epochs=100,)

#4. 평가, 예측
result = model.evaluate ([x1_datasets, x2_datasets, x3_datasets], [y1, y2])
print ("loss :", result)

# x1_pred = np.array ([range(100, 106), range(400, 406)]).T 
# x2_pred = np.array ([range(200, 206), range(510, 516), range(249, 255)]).T
# x3_pred = np.array ([range(100,106), range(400,406), range(177,183), range(133,139)]).T

# 예측값까지 완성