#acc 0.6 이상
"""
*Embedding +LSTM + bidirectional
loss: 0.32271867990493774
acc: 0.8718400001525879

*Embedding + GRU + Bidirectional
loss: 0.31196287274360657
acc: 0.8704000115394592

*Embedding + Flatten + DNN
loss: 0.30856332182884216
acc: 0.8703200221061707
걸린 시간 : 3.21 s
"""

import time
import datetime
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GRU, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

save_path = "./_save/keras62_imdb/"

#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=20000, #단어사전의 개수, 빈도수의 개수가 높은 단어 순으로 1000개 뽑겠다
    # maxlen = 100, # 단어갯수의 최대 길이 제한
)
# print (x_train.shape, y_train.shape) #(25000,) (25000,)
# print (x_test.shape, y_test.shape) #(25000,) (25000,)
# print (np.unique(y_train)) #[0 1]
# print ("imdb의 최대 길이 :", max (len (i) for i in x_train)) # 2494
# print ("imdb의 최소 길이 :", min (len (i) for i in x_train)) # 11
# print ("imdb의 평균 길이 :", round(sum(map( len, x_train))/ len(x_train), 2)) # 238.71

## 전처리
padded_x_train = pad_sequences(x_train,
                        maxlen = 200, #최대 길이 결정
                        padding='pre', #padding='post'
                        truncating ='pre' # 자르는 건 default가 앞
                        )
padded_x_test = pad_sequences(x_test,
                        maxlen = 200, #최대 길이 결정
                        padding='pre', #padding='post'
                        truncating ='pre' # 자르는 건 default가 앞
                        )
print (padded_x_train.shape) #(8982, 300)
print (padded_x_test.shape) #(2246, 300)

#2. 모델 구성
## 2-1 Embedding +LSTM + bidirectional
# model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=128, mask_zero=True))
# model.add(Bidirectional(GPU(64, dropout=0.2)))
# model.add(Dropout(0.4))
# model.add(Dense(1, activation='sigmoid'))

## 2-2 Embedding + Bidirectional GRU
# model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=128, mask_zero=True))
# model.add(Bidirectional(GRU(64, dropout=0.2)))
# model.add(Dropout(0.4))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))

## Embedding + Flatten + DNN
model = Sequential()
model.add(Input(shape=(200,)))
model.add(Embedding(input_dim=20000, output_dim=128))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=3,
    factor=0.5,
    min_lr=1e-6,
    verbose=1
)

start_time = time.time()
model.fit(
    padded_x_train, y_train,
    epochs=50,
    batch_size=1000,
    validation_split=0.1,
    callbacks=[es, rlr]
)
end_time = time.time()

#4. 예측, 평가
loss, acc = model.evaluate(padded_x_test, y_test)
print('loss:', loss)
print('acc:', acc)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')