# # https://dacon.io/competitions/open/235576

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error

# #1.데이터

# path = "./_data/ddarung/"

# #pandas 넘파이로 이루어진 사이킷만큼 강력한 놈
# import pandas as pd

# train_csv = pd.read_csv(path + 'train.csv') #read_csv는 첫번째 라인은 통상적으로 기본 컬럼명으로 인식해서 제외한다.

# print (train_csv) # [1459 rows x 11 columns] -> id열 포함 // 인덱스는 데이터가 아니다.  인덱스를 빼야 한다.

# # pandas 에는 첫번째 인덱스 데이터를 처리하지 말라고 하는 파라미터가 있다. 그 파라미터가 index_col
# train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print (train_csv) #[1459 rows x 10 columns] -> id 미포함 
# #train 데이터를 보면 submission 데이터 있는 count 칼럼이 train 데이터에 있다 이걸 분리해야 한다.

# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print (test_csv) #[715 rows x 9 columns]

# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# print (submission) #[715 rows x 1 columns] #데이터가 비어있으면 NaN 표시한다 (결측치)

# print(train_csv.shape) #(1459, 10) 
# print(test_csv.shape) #(715, 9)
# print(submission.shape) #(715, 1)
# #train.csv 만으로 훈련 / test.csv와 submission.csv는 제출용
# # 훈련을 위해 train.csv 를 x와 y로 분리해야 함.

# print (train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
# #       dtype='str')

# print (train_csv.info())
# print (test_csv.info())

# # exit ()
# ############################# 결측치 처리방법 많지만 오늘은 1. 삭제 #############################
# train_csv = train_csv.dropna() #dropna -> 결측치를 가진 행 제거
# print (train_csv) #[1328 rows x 10 columns]
# print (train_csv.info())

# # **train_csv를 x와 y로 분리**
# x = train_csv.drop(['count'], axis=1) # axis 는 축. axis=0 행(row) | axis=1 열(column) 그러므로 이 문장은 열 삭제
# print (x) #[1328 rows x 9 columns]

# y = train_csv['count']
# print (y.shape) #(1328,)

# x_test = test_csv
# y_test = submission

# #2.모델구성
# model = Sequential()
# model.add(Dense(10, input = 9))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))


# #3.컴파일, 훈련
# model.compile (loss = 'mse', optimizer = 'adam')
# model.fit (x, y, epochs = 100, batch_size = 10)

# #4.평가, 예측
# def RMSE (x_test, y_test): #RMSE 함수 정의
#     return np.sqrt(mean_squared_error(x_test, y_test))

# rmse = RMSE(x_test, y_test)
# print("RMSE : ", rmse)


