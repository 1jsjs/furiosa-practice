# https://dacon.io/competitions/open/235576

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = "./_data/ddarung/" #상대경로
# path = "c:\study\_data\ddarung/" # 절대경로
# path = "c:\\study\\_data\\ddarung/" 
# path = "c:/study/_data/ddarung/"
# path = "c://study//_data//ddarung/"

#인덱스는 데이터가 아니다
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv.shape) #(1459, 10) 
# print(test_csv.shape) #(715, 9)
# print(submission.shape) #(715, 1)

# #train.csv 만으로 훈련 / test.csv와 submission.csv는 제출용
# # 훈련을 위해 train.csv 를 x와 y로 분리해야 함.
# 훈련 방식은 x는 y야 반복 훈련

train_csv = train_csv.dropna() 

# print (train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
# #       dtype='str')

# **train_csv를 x와 y로 분리**
x = train_csv.drop(['count'], axis=1) #count라는 칼럼을 탈락시킴 //  axis=0 행(row) | axis=1 열(column) 그러므로 이 문장은 열 삭제
y = train_csv['count']

# train 데이터를 학습용과 평가용으로 다시 한번 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=12345)

# submission 작업
print (test_csv.shape) #(715, 9)
print (test_csv.info()) #결측치 발생!!!!!!

"""
저번에 사용했던 결측치 제거 방식은 안된다 // 1.제거 2.평균값넣기
이번엔 평균값으로

test_csv는 pandas 데이터 프레임 형태
"""
#NULL 값들을 test_csv 데이터 평균값에 넣기 
test_csv = test_csv.fillna (test_csv.mean())
print (test_csv.info()) #결측치 처리 완료
print (test_csv.shape) #(715, 9)

#exit()

#2.모델구성
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs =500, batch_size = 10)


#4.평가, 예측
print("========================================")
y_predict = model.predict(x_test) #지표가 잘 맞는지 평가

r2 = r2_score(y_test, y_predict)
print ("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print ("MSE: ", mse)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print ("RMSE: ", rmse)

##################################submission.csv 만들기 // count 컬럼에 값을 넣어준다.
# print (submission)

#submission['count'] = 모델.predict 한 결과
y_submit = model.predict (test_csv)
submission['count'] = y_submit
# print (submission)
# print (submission.shape) #(715, 1)
submission.to_csv(path + 'submit/' + "submit_0904_1342.csv")