# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1.데이터
path = "./_data/kaggle_santander/" #상대경로 맨 마지막에 / 빼지말기
# path = "C:\\study\\_data\\kaggle_santander"

train_csv = pd.read_csv (path + 'train.csv', index_col=0)
test_csv = pd.read_csv (path + 'test.csv',  index_col=0)
submission_csv = pd.read_csv (path + 'sample_submission.csv', index_col=0)

# print (train_csv.shape) #(200000, 201)
# print (test_csv.shape) #(200000, 200)
# print (submission_csv.shape) #(200000, 1)
#결측치 확인
# print (train_csv.isna().sum())
# print ("==")
# print (test_csv.isnull().sum())
# print ("==")
# print (submission_csv.isna().sum())
# print ("==")

#1-1. x와 y 분리
x = train_csv.drop(['target'], axis = 1)
y = train_csv['target']
# print (x.shape, y.shape) #(200000, 200) (200000,)
# print (np.unique(y, return_counts=True)) #(array([0, 1]), array([179902,  20098]))

#1-2. train 데이터를 다시 한번 분리
x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.9,
                                                     random_state=42112,
                                                     stratify = y)

#2.모델구성
model = Sequential()
model.add (Dense(256, input_dim = 200, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일, 훈련
model.compile (loss = 'binary_crossentropy',
               metrics=['accuracy'],
               optimizer = 'adam')

es = EarlyStopping (
    monitor = 'val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()

model.fit (x_train, y_train,
           epochs = 1000,
           batch_size = 500,
           validation_split = 0.2,
           callbacks = [es])

end_time = time.time()

#4. 평가, 예측
print ("걸린 시간:", round(end_time-start_time, 2))
y_pred = model.predict (x_test)
print (y_pred)
y_submit = model.predict (test_csv)
y_submit = np.round (y_submit)

submission_csv['target'] = y_submit
submission_csv.to_csv(path + 'submit/' + 'submit_0908_1740.csv')