# https://dacon.io/competitions/open/235576

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = "./_data/ddarung/"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)


train_csv = train_csv.dropna() 

# **train_csv를 x와 y로 분리**
x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']

# train 데이터를 학습용과 평가용으로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)


#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 10, batch_size = 10)


#4.평가, 예측
print("========================================")
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error

r2 = r2_score(y_test, y_predict)
print ("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print ("MSE: ", mse)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print ("RMSE: ", rmse)


# 따릉이에서 헷갈린 이유는 test라는 이름이 두 번 다른 뜻으로 쓰였기 때문
# train.csv  ← 정답 count가 있는 원본 데이터
#    │
#    ├─ x, y로 분리
#    │    ├─ x: 문제(9개 날씨·시간 특성)
#    │    └─ y: 정답(count)
#    │
#    └─ 다시 학습용 / 평가용으로 분리
#         ├─ x_train, y_train → #3 model.fit()에 사용
#         └─ x_val, y_val     → #4 내 모델 성능 평가에 사용

# test.csv                 → 정답이 없으므로 평가 불가, 예측만 함
# submission.csv           → test.csv 예측값을 담아 제출하는 양식


# x_train, x_val, y_train, y_val
# 대회와 실무 모두 보통 이 원리를 씁니다.
# - 정답 있는 과거 데이터 → 학습용과 검증용으로 나눔
# - 정답 없는 새 데이터 → 예측
# - 실제 정답이 나중에 생기면 → 성능 평가
# 네가 몰랐던 건 x/y 분리가 아니라, 평가할 정답을 확보하려고 train 데이터에서 한 번 더 나눈다는 단계임