#keras28_Scaler03_boston.py copy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


import numpy as np
import time

# path = "./_save/keras33_boston/"


#1.데이터
#이미 데이터셋에서 train, test가 분리되어 있음
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print (x_train.shape, x_test.shape) #(404, 13) (102, 13)
print (y_train.shape, y_test.shape) #(404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_test = scaler.transform (x_test)


#2.모델구성
#2. 함수형 모델 구성
input1 = Input(shape=(13,))
dense1 = Dense(100)(input1)
dense2 = Dense(256, activation='relu')(dense1)
dropout3 = Dropout(0.2)(dense2)
dense4 = Dense(100, activation='relu')(dropout3)
dropout5 = Dropout(0.2)(dense4)
dense6 = Dense(500, activation='relu')(dropout5)
dropout7 = Dropout(0.2)(dense6)
dense8 = Dense(256, activation='relu')(dropout7)
dropout9 = Dropout(0.2)(dense8)
dense10 = Dense(150, activation='relu')(dropout9)
dropout11 = Dropout(0.2)(dense10)
dense12 = Dense(150, activation='relu')(dropout11)
dropout13 = Dropout(0.2)(dense12)
dense14 = Dense(50, activation='relu')(dropout13)
output1 = Dense(1)(dense14)
model = Model(inputs=input1, outputs=output1)
model.summary()
#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# path = "./_save/keras33_boston/"
# filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
# filepath = "".join([path, "k33_", date, filename])
# mcp = ModelCheckpoint ( 
#     monitor='val_loss',
#     mode='auto',
#     save_best_only = True,
#     filepath =filepath,
#     verbose=1,
# )
start_time = time.time()
model.fit (x_train, y_train, epochs = 1000, batch_size =10, 
           validation_split = 0.2,
           callbacks = [es])
end_time = time.time()

print ("=================================================")
#4.평가, 예측
loss = model.evaluate (x_test, y_test, ) #default batch_size = 32 //tensorflow의 evaluate
# evaluate 를 풀어서 쓴다면?
#  y(예측값) = w * x + b
#             w * x_test + b
#      loss = Σ(y_i(test) - ŷ_i)^2 / n
# loss 할 때 x_test, y_test 둘다 쓴다.

#y^- = w(마지막 epoch 에서 나온 가중치 값으로) * x_test + b
print ('loss(mse) : ', loss) #loss :  28.8504581451416

y_predict = model.predict (x_test) #y 예측값이라서 y_predict라고 써도 됨 (이해를 돕기 위해)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict) # 인풋과 아웃풋을 다 알아야 한다.
print ("r2 :", r2)

#사이킷런에서는 원값과 테스트값만 넣으면 된다!

mse = mean_squared_error(y_test, y_predict) #사이킷런의 mse
print("mse : ", mean_squared_error)

def RMSE (y_test, y_predict): #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print ("걸린 시간:", round(end_time-start_time, 2))


"""
9.14
loss(mse) :  16.88728904724121
r2 : 0.7971346927338385
mse :  <function mean_squared_error at 0x0000018EA6C980E0>
RMSE :  4.10941466732028
걸린 시간: 11.17



loss(mse) :  19.208751678466797
r2 : 0.7692471900655737
mse :  <function mean_squared_error at 0x000001A3EBF47740>
RMSE :  4.382779022001862
걸린 시간: 8.58

"""