# keras30_ModelCheckPoint2_load.py copy


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import time

path = "./_save/keras30/"

#1.데이터
datasets = fetch_california_housing ()
x = datasets.data
y = datasets.target

# datasets 을 x와 y로 분리
x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.7,
                                                    #  shuffle=True, 
                                                     random_state=42)
# 원 데이터 전체를 스케일러 하면 안 된다. train만 scaler한다.

# scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit (x_train) #x값은 모두 민맥스 스케일러 할 준비를 해라
x_train = scaler.transform (x_train) #변환시키기
x_test = scaler.transform (x_test) #변환시키기

print (x)
print (np.min(x_train), np.max(x_train)) #0.0 1.0000000000000004
print (np.min(x_test), np.max(x_test)) #-0.0010638297872338498 1.333173652694611
# exit()

#2.모델구성 (input_dim = 8 / 행무시 열우선)
model = Sequential()
model.add (Dense(10, input_dim = 8))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련 (loss mse, op adam /훈련은 x와y train으로 / 배치 모르면 당분간은 디폴트로 )
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping (
    monitor= 'val_loss', #기준을 선언
    mode= 'min', #어떤 값을 찾을까? 긴가민가 하면 auto 하면 됨
    patience= 30, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
    verbose = 1, 
)

# 몇 에포에 어떤 loss값인지 다 나타나면 좋겠다는 생각 그러면 작업할 때 편하니까 > 30-3 에서 어떻게 하는지 알려줌

################################### mcp 세이브 파일 만들기 끝 ###################################
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = "./_save/keras30/"
filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path, "k30_", date, filename])
# 파일명 예
# "./_save/keras30/" + "k30_", 0914_1147, 에포수-val_loss의 소수 4번째자리까지.keras
mcp = ModelCheckpoint ( 
    monitor='val_loss',
    mode='auto',
    save_best_only = True,
    filepath =filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit (x_train, y_train,
                  epochs = 1000, batch_size = 32,
                  validation_split = 0.2,
                  callbacks = [es, mcp],
                  verbose=1
                  )
end_time = time.time()


#4.평가, 예측 (evaluate 는 test로 / predict 까지는 보류 / 판단은 evaluate의 loss 값)
y_predict = model.predict (x_test)
mse = mean_squared_error(y_test, y_predict)
print ("MSE: ", mse)

r2 = r2_score(y_test, y_predict)
print ("r2 : ", r2)

loss = model.evaluate (x_test, y_test)
print ('loss :', loss) #loss : 0.6562087535858154
print ("걸린 시간:", round(end_time-start_time, 2))

"""
MSE:  0.2693821316453831
r2 :  0.7947300173106727
loss : 0.269382119178772
걸린 시간: 53.08
"""