import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

#원핫 때려주고 np유니 확인 한번 해주시고 모델에서는 마지막 softmax 평가예측할땐 categorical~ 마지막np.round > np.argmax

#1. 데이터
path = "./_data/kaggle_santander/"

train_csv = pd.read_csv (path + 'train.csv', index_col=0)
test_csv = pd.read_csv (path + 'test.csv', index_col=0)
submission_csv = pd.read_csv (path + 'sample_submission.csv', index_col=0)
# print (train_csv.shape) #(200000, 201)
# print (test_csv.shape) #(200000, 200)
# print (submission_csv.shape) #(200000, 1)
# # 결측치 확인
# print (train_csv.isna().sum())
# print ("==")
# print (test_csv.isnull().sum())
# print ("==")
# print (submission_csv.isna().sum())
# print ("==")

#1-1. x와 y 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
# print (x.shape, y.shape) #(200000, 200) (200000,)
# print (np.unique(y, return_counts=True)) #(array([0, 1]), array([179902,  20098]))

y = pd.get_dummies(y, dtype=int)
# print (y)
# print (y.shape) #(200000, 2)

#1-2. train 데이터를 다시 한번 분리
x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     stratify=y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_test = scaler.transform (x_test)

#2. 모델 구성
model = Sequential()
model.add (Dense(256, input_dim = 200, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# print (x_train.shape, x_test.shape) #(160000, 200) (40000, 200)
# print (y_train.shape, y_test.shape) #(160000, 2) (40000, 2)

#3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs = 1000, batch_size = 1000,
           verbose=1, 
           validation_split=0.2,
           callbacks=[es],
           )
# error 1 : ValueError: Arguments `target` and `output` must have the same rank (ndim). Received: target.shape=(None,), output.shape=(None, 3)
# dim 나오면 칼럼 생각하면 된다. // 원래 dense를 3을 줬는데 1로 바꾸기 > 되긴 되는데 
# UserWarning: You are using a softmax over axis -1 of a tensor of shape (32, 1). This axis has size 1. 
# #The softmax operation will always return the value 1, which is likely not what you intended. Did you mean to use a sigmoid instead?
end_time = time.time()

#4. 평가, 예측
# results = model.evaluate (x_test, y_test)
# print ('loss :', results[0])
# print ('acc :', round(results[1], 2))

# y_pred = model.predict(x_test)
# # print (y_pred)
# y_pred = np.argmax(y_pred, axis=1)
# # print (y_pred)
# y_test = np.argmax(y_test, axis=1)
# # print (y_test)

# accuracy_score = accuracy_score(y_test, y_pred)
# print ('acc_score : ', accuracy_score)
# print ('걸린 시간 : ', round(end_time - start_time, 2), 's')
# """
# loss : 0.24460369348526
# acc : 0.91
# acc_score :  0.910725
# 걸린 시간 :  116.03 s
# """
# y_submit = model.predict (test_csv)
# y_submit = np.argmax (y_submit)

# submission_csv['target'] = y_submit
# submission_csv.to_csv (path + 'submit/' + 'submit_0910_1046.csv')
print ("걸린 시간:", round(end_time-start_time, 2))
y_pred = model.predict (x_test)
print (y_pred)
y_submit = model.predict (test_csv)
y_submit = np.argmax (y_submit, axis=1)

submission_csv['target'] = y_submit
submission_csv.to_csv(path + 'submit/' + 'MaxAbsScaler_submit_0910_1713.csv')


# 26.09.10 기준 minmax scaler 적용
"""
왜 자꾸 0.5가 나오지? csv에 값도 이상함
"""