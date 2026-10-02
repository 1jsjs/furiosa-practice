#다중 분류다!!!
# acc 0.67 이상
"""loss : 1.5771799087524414
acc : 0.6108637452125549
accuracy score : 0.6108637577916296
걸린 시간 : 815.12 s"""
import time
import datetime
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

save_path = "./_save/keras61_reuters/"

#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 10000, #단어사전의 개수, 빈도수의 개수가 높은 단어 순으로 1000개 뽑겠다
    # maxlen = 100, # 단어갯수의 최대 길이 제한
    test_split = 0.2,
)
# print (x_train.shape, y_train.shape) #(8982,) (8982,)
# print (x_test.shape, y_test.shape) #(2246,) (2246,)
# print (np.unique(y_train))
"""
46개의 labels
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
"""
# print (type(x_train)) #<class 'numpy.ndarray'>
# print (type(x_train[0])) #<class 'list'>
# print (len(x_train[0])) 

# print ("뉴스 기사의 최대 길이 :", max (len (i) for i in x_train))
# print ("뉴스 기사의 최소 길이 :", min (len (i) for i in x_train))
# print ("뉴스 기사의 평균 길이 :", round(sum(map( len, x_train))/ len(x_train), 2))

## 전처리
padded_x_train = pad_sequences(x_train,
                        maxlen = 300, #최대 길이 결정
                        padding='pre', #padding='post'
                        truncating ='pre' # 자르는 건 default가 앞
                        )
padded_x_test = pad_sequences(x_test,
                        maxlen = 300, #최대 길이 결정
                        padding='pre', #padding='post'
                        truncating ='pre' # 자르는 건 default가 앞
                        )
# print (padded_x_train.shape) #(8982, 300)
# print (padded_x_test.shape) #(2246, 300)

## one-hot (y값)
y_train = to_categorical(y_train, num_classes=46)
y_test = to_categorical(y_test, num_classes=46)
# print(y_train.shape, y_test.shape)  # (8982, 46) (2246, 46)

#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(Bidirectional(LSTM(64, dropout=0.2)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile (loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['acc'],)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=200,
    restore_best_weights=True,
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=7,
    factor=0.5,
    min_lr=1e-6,
    verbose=1
)

date = datetime.datetime.now().strftime("%m%d_%H%M")
filepath = save_path + f"k61_{date}_{{epoch:04d}}-{{val_loss:.4f}}.keras"

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

start_time = time.time()
model.fit(padded_x_train, y_train, epochs=5000, batch_size=64, shuffle=False, verbose=1, validation_split=0.1, callbacks=[es, rlr, mcp])
end_time = time.time()


#4. 예측, 평가
loss = model.evaluate (padded_x_test, y_test)
y_pred = model.predict (padded_x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)
y_test = np.argmax(y_test, axis=1) #.reshape(-1,1)
acc_score = accuracy_score(y_test, y_pred)

print ('=============================')
print ('loss :', loss[0])
print ('acc :', loss[1])
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')