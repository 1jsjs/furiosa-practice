import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

#원핫 때려주고 np유니 확인 한번 해주시고 모델에서는 마지막 softmax 평가예측할땐 categorical~ 마지막np.round > np.argmax


savepath = "./_save/keras41/"

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
model = load_model(savepath + "keras41k41_0914_14310006-0.238372.keras")

# print (x_train.shape, x_test.shape) #(160000, 200) (40000, 200)
# print (y_train.shape, y_test.shape) #(160000, 2) (40000, 2)

#3. 컴파일, 훈련

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
y_pred = model.predict (x_test)
print (y_pred)
y_submit = model.predict (test_csv)
y_submit = np.argmax (y_submit, axis=1)

submission_csv['target'] = y_submit
submission_csv.to_csv(path + 'submit/' + 'MCP_save_submit_0914_1430.csv')


# 26.09.10 기준 minmax scaler 적용
"""
왜 자꾸 0.5가 나오지? csv에 값도 이상함
"""


"""
[[0.99096024 0.00903975]
 [0.98431814 0.01568192]
 [0.46712437 0.5328757 ]
 ...
 [0.990265   0.00973499]
 [0.9840449  0.01595509]
 [0.8917706  0.10822935]]

 
[[0.99096024 0.00903975]
 [0.98431814 0.01568192]
 [0.46712437 0.5328757 ]
 ...
 [0.990265   0.00973499]
 [0.9840449  0.01595509]
 [0.8917706  0.10822935]]

"""