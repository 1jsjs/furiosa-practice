#https://www.kaggle.com/competitions/bike-sharing-demand/data
# keras14_kaggle_bike1.py copy

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = "./_data/kaggle_bike-sharing-demand/" #상대경로 맨 마지막에 / 빼지말기

#1-1.인덱스는 데이터가 아니다.
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# print (train_csv.shape) # (10886, 11)
# print (test_csv.shape) # (6493, 8)
# print (submission_csv.shape) # (6493, 1)
################ x, y 분리 ################
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']
# print (x) #[10886 rows x 8 columns]
# print (y.shape) # (10886,)
"""
casual 과 registered 칼럼들은 test는 없어서 그냥 뺌
"""

#train 데이터를 학습용과 평가용으로 다시 한번 분리
x_train, x_val, y_train, y_val = train_test_split (x, y, train_size=0.9, random_state=42)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_val = scaler.transform (x_val)


x_train = x_train.reshape (-1,8,1,1)
x_val = x_val.reshape (-1,8,1,1)

print (x_train.shape)
print (x_val.shape)
#2.모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(8,1,1), padding='same')) 
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(1)) 

#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
hist = model.fit (x_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.3)

#4.평가, 예측
print("========================================")
y_predict = model.predict (x_val)

r2 = r2_score(y_val, y_predict)
print ("r2 : ", r2)

mse = mean_squared_error(y_val, y_predict)
print ("MSE: ", mse)

def RMSE (y_val, y_predict):
    return np.sqrt(mean_squared_error(y_val, y_predict))

rmse = RMSE(y_val, y_predict)
print ("RMSE: ", rmse)

y_submit = model.predict (test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv(path + 'submit/' + 'RobustScaler_ES_val_submit_0911_1649.csv')

"""
5차 시도
random_state = 42
train_size = 0.90
epoch = 1000
batch_size = 50

r2 :  0.23310333490371704
MSE:  25383.892578125
RMSE:  159.32323301428767

1.50930

마지막 y값을 건드려서 값이 더 올라간듯
"""

# 26.09.10 기준 scaler 적용
"""
r2 :  -1.0889439582824707
MSE:  69142.9921875
RMSE:  262.95055084083776
"""

# 26.09.11 기준 standard scaler 적용
"""
r2 :  0.3290058970451355
MSE:  22209.56640625
RMSE:  149.02874355724134
"""

"""
# 26.09.11 기준 MaxAbsScaler 적용
r2 :  0.33581870794296265
MSE:  21984.06640625
RMSE:  148.2702478795055
"""

"""
# 26.09.11 기준 RobustScaler 적용
r2 :  0.33581870794296265
MSE:  21984.06640625
RMSE:  148.2702478795055
"""
"""
0917 dnn cnn
r2 :  -0.9384737014770508
MSE:  64162.50390625
RMSE:  253.30318574042846
"""