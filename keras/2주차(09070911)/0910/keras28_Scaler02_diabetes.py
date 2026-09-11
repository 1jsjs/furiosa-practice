# keras12_R2_RMSE_03_diabetes.py copy


from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time

#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print (x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.80, random_state=333)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform (x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform (x_test)

#2.모델구성 (input_dim = 10, 행무시 열우선)
model = Sequential()
model.add(Dense(300, input_dim = 10))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))


#3.컴파일, 훈련
import time
model.compile (loss = 'mse', optimizer = 'adam')
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs = 1000,
           batch_size = 3, 
           validation_split = 0.2,
           callbacks = [es])
end_time = time.time()

#4.평가, 예측
loss = model.evaluate (x_test, y_test)
print ('loss : ', loss) #loss :  2690.05908203125

y_predict = model.predict (x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score (y_test, y_predict)
print ("r2 :", r2) #r2 : 0.5057520187712794

loss = model.evaluate (x_test, y_test)
print ('loss :', loss)
print ("걸린 시간:", round(end_time-start_time, 2))


#R2 기준 0.62 이상
#r2 : 0.5133829603918585

# 26.09.10 기준 minmax scaler 적용
"""
r2 : 0.42872237043373007
loss : 3028.859619140625
걸린 시간: 7.23
"""

"""
# 26.09.11 기준 standard scaler 적용
r2 : 0.3609701747001729
loss : 3388.0751953125
걸린 시간: 10.19
"""

"""
# 26.09.11 기준 MaxAbsScaler 적용
loss :  4133.71923828125
r2 : 0.22033324088500428 
loss : 4133.71923828125
걸린 시간: 13.07
"""

"""
# 26.09.11 기준 RobustScaler 적용
loss :  3312.933349609375
r2 : 0.37514282411740596
loss : 3312.933349609375
걸린 시간: 8.08
"""

# import matplotlib.pyplot as plt

# print ("==============================================hist=====================================================")
# print (hist) #지금 hist는 랩핑되어 있는 상태
# print ("==============================================hist.history=====================================================")
# print (hist.history) #지금 hist는 랩핑되어 있는 상태, epoch 횟수만큼 저장되어 있음 fit 함수는 loss와 val loss 값을 반환하고 있었다.
# # ai 할 때는 리스트(두 개 이상은 리스트) 와 딕셔너리(key-value는 딕셔너리) 값을 많이 쓴다.
# # 이것 가지고 시각화하면 더 이해를 잘 할 수 있게 되겠지?
# print ("==============================================loss=====================================================")
# print (hist.history['loss'])
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
# plt.title('diabetes')
# plt.ylabel('loss')
# plt.grid()
# plt.show()