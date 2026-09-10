#https://www.kaggle.com/competitions/bike-sharing-demand/data
# keras14_kaggle_bike1.py copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_val = scaler.transform (x_val)

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

submission_csv.to_csv(path + 'submit/' + 'scaler_ES_val_submit_0910_1702.csv')

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


# import matplotlib.pyplot as plt

# print ("==============================================hist=====================================================")
# print (hist) #지금 hist는 랩핑되어 있는 상태
# print ("==============================================hist.history=====================================================")
# print (hist.history) #지금 hist는 랩핑되어 있는 상태, epoch 횟수만큼 저장되어 있음 fit 함수는 loss와 val loss 값을 반환하고 있었다.
# # ai 할 때는 리스트(두 개 이상은 리스트) 와 딕셔너리(key-value는 딕셔너리) 값을 많이 쓴다.
# # 이것 가지고 시각화하면 더 이해를 잘 할 수 있게 되겠지?
# print ("==============================================loss=====================================================")
# print (hist.history['loss'], 'epochs : ', len(hist.history['loss']))
# print ("==============================================val_loss=====================================================")
# print (hist.history['val_loss'])

# print ("==============================================시각화하기=====================================================")


# plt.rcParams["font.family"] = "Malgun Gothic"
# plt.rcParams["axes.unicode_minus"] = False

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.xlabel('epoch')
# plt.title('캐글 자전거 데이터')
# plt.ylabel('loss')
# plt.grid()
# plt.show()