# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# 10분 간격 7년치 시계열 데이터
# wd 를 y로 // 144개가 time steps 로 하고 하루치 wd를 맞춰라
# 맞추기 : 2016년 12월 31일 00시 10분부터
# 2017.01.01 00:00:00깓지 데이터 144개 (훈련에 쓰지 않음)
# 훈련 데이터에는 12월 31일 데이터는 들어가면 안 된다. (뒤에서부터 144개 자르기)
# 144개가 1일 

#0930 수정 : y를 T (degC) 로
"""
k58_0930_1510_0078-0.0348.keras
R2: 0.5515135638792179
MSE: 4.8863121985150855
RMSE:  2.2105004407407582
걸린 시간 :  1412.15 s

k58_0930_1535_0122-0.0343.keras
R2: 0.6775548099357175
MSE: 3.5130780747609993
RMSE:  1.8743206968822061
걸린 시간 :  1061.14 s
"""
#====================================================
"""
k58_0929_1706_0003-0.5495.keras
R2: -0.12322989753173386
MSE: 3257.7342141688314
걸린 시간 :  243.14 s

k58_0929_1726_0037-0.5600.keras
R2: -0.06665349810689425
MSE: 3093.644144517196
걸린 시간 :  785.42 s
"""
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import time
import datetime
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
data_path = "./_data/kaggle_jena/"
save_path = "./_save/keras58_kaggle_jena/"
np_path = './_data/jena_npy/'           


datasets = pd.read_csv(data_path + 'jena_climate_2009_2016.csv', index_col=0)

y_cor = datasets[-144:]["T (degC)"] #예측치 정답 데이터

x_data = datasets[:-288].drop("T (degC)", axis=1)
y_data = datasets[144:-144]["T (degC)"]
"""
y_cor에 예측 정답 144개를 따로 보관하고, 학습용 x_data와 y_data는 144행 차이를 두는 구성
이 CSV에서는 y_cor가 말씀한 대로 2016-12-31 00:10부터 2017-01-01 00:00까지의 행을 잡음
"""
# print (x_data.shape, y_data.shape) #(420263, 13) (420263,)

#============================================================================== 선생님 코드

x_scaler = RobustScaler()
y_scaler = RobustScaler()

x_data = x_scaler.fit_transform(x_data.to_numpy())
y_data = y_scaler.fit_transform(y_data.to_numpy().reshape(-1, 1))[:, 0]

def split_x (dataset, size):
    aaa = []
    for i in range(len(dataset) -size +1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)
# x는 (N,144,13) // y는 (N,144)
# 그리고 이제 x와 y로 나누기 

# pandas로 불러온 csv 데이터를 numpy로 바꾸기 .to_numpy() !!
x = split_x(x_data, 144)
y = split_x(y_data, 144)
print (x.shape, y.shape) #(420120, 144, 13) (420120, 144)

np.save(np_path + 'x_train.npy', x)
np.save(np_path + 'y_train.npy', y)

y = y.reshape(y.shape[0], 144, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(32, input_shape=(144, 13), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(144))     

#3. 컴파일, 훈련
learning_rate = 0.003

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
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
filepath = save_path + f"k58_{date}_{{epoch:04d}}-{{val_loss:.4f}}.keras"

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

start_time = time.time()
model.fit(x, y, epochs=5000, batch_size=2000, shuffle=False, verbose=1, validation_split=0.1, callbacks=[es, rlr, mcp])
end_time = time.time()

# 4. 평가
def RMSE (y_val, y_predict):
    return np.sqrt(mean_squared_error(y_val, y_predict))

x_predict = datasets[-288:-144].drop(["T (degC)"], axis=1).to_numpy()
x_predict = x_scaler.transform(x_predict).astype(np.float32)
x_predict = x_predict.reshape(1, 144, 13)

y_cor_scaled = y_scaler.transform(
    y_cor.to_numpy().reshape(-1, 1)
)[:, 0].astype(np.float32).reshape(1, 144)

print("평가:", model.evaluate(x_predict, y_cor_scaled, verbose=1))

y_predict_scaled = model.predict(x_predict, verbose=0)[0]  # (144,)
y_predict = y_scaler.inverse_transform(
    y_predict_scaled.reshape(-1, 1)
)[:, 0]  # (144,)


rmse = RMSE(y_cor.to_numpy(), y_predict)

print('12월 31일 실제값:', y_cor.to_numpy())
print('12월 31일 예측값:', y_predict)
print('R2:', r2_score(y_cor.to_numpy(), y_predict))
print('MSE:', mean_squared_error(y_cor.to_numpy(), y_predict))
print ("RMSE: ", rmse)
print ('걸린 시간 : ', round(end_time - start_time, 2), 's')