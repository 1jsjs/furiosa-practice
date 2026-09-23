
# 실습 acc 0.67이상
"""
epochs=100, batch_size=128, verbose=1, validation_split=0.1
loss : 3.7106895446777344
acc : 0.12800000607967377
accuracy score : 0.128
걸린 시간 : 542.4 s

epochs=100, batch_size=256, verbose=1, validation_split=0.1
loss : 3.7265281677246094
acc : 0.09300000220537186
accuracy score : 0.093
걸린 시간 : 779.3 s


0917 - 맥스풀링 추가
loss : 3.6535484790802
acc : 0.09769999980926514
accuracy score : 0.0977
걸린 시간 : 367.06 s

0917- GlobalAveragePooling(각 채널 전체의 평균을 내서 Flatten처럼 1차원으로 변환)


# optimizer - learning_rate = 0.0001
loss : 1.8447939157485962
acc : 0.5430999994277954
accuracy score : 0.5431
걸린 시간 : 960.6 s
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print (x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print (x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
# print (np.max(x_train), np.min(x_train)) #255 0
# print (np.max(x_test), np.min(x_test)) #255 0
# print (np.max(y_train), np.min(y_train)) #9 0
# print (np.max(y_test), np.min(y_test)) #9 0

######## 1. 데이터
########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print (x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print (x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(x_train.min(), x_train.max()) #-1.0 1.0

# OneHot
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train) 
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)
# print (y_train.shape, y_test.shape) #(50000, 100) (10000, 100)


######## 2. 모델구성
model = Sequential()

# 32 x 32 x 3
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu',input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

# 16 x 16 x 64
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
model.add(Dropout(0.3))


# 8 x 8 x 128
model.add(Conv2D(256, (3, 3),strides=(1, 1),padding='same',activation='relu'))

model.add(Conv2D(256, (3, 3),strides=(1, 1),padding='same',activation='relu'))

model.add(Conv2D(256, (3, 3),strides=(1, 1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
model.add(Dropout(0.35))


# 4 x 4 x 256
model.add(Conv2D(512, (3, 3),strides=(1, 1),padding='same',activation='relu'))

model.add(Conv2D(512, (3, 3),strides=(1, 1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
model.add(Dropout(0.4))


# 2 x 2 x 512
# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(units=100,activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001

model.compile (loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
               metrics=['acc'])
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs=200, batch_size=128, verbose=1, validation_split=0.1, callbacks =[es])
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)
y_test = np.argmax(y_test, axis=1) #.reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')