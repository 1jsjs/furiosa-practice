#https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

#1.데이터
path = "./_data/kaggle_bike-sharing-demand/" #상대경로 맨 마지막에 / 빼지말기

#1-1.인덱스는 데이터가 아니다.
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# print (train_csv.shape) # (10886, 11)
# print (test_csv.shape) # (6493, 8)
# print (submission_csv.shape) # (6493, 1)
"""
train_csv로만 훈련한다.
훈련을 위해 train_csv를 x와 y로 분리해야 함
train.csv, test.csv, sampleSubmission.csv 셋다 결측치와 이상치가 없다.
확인하는 방법은 info()이다.

describe() 함수를 쓰면 더 자세한 정보가 나온다.

####################결측치 확인#####################
pandas 의 isna() 는 NaN 값이 있는지 볼 수 있음 isnull() 함수도 가능
isna().sum() 을 하면 결측치의 개수를 볼 수 있다.
"""
# print (train_csv.info()) # (10886, 11)
# print (train_csv.info()) # (10886, 11)
# print (train_csv.info()) # (10886, 11)
# print (train_csv.describe())
# print (train_csv.describe())
# print (test_csv.describe())
# print (submission_csv.describe())

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

#2.모델구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 1000, batch_size = 50)

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

submission_csv.to_csv(path + 'submit/' + 'submit_0904_1732.csv')

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