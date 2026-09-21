# validation은 훈련에 참여를 안 한다.
# train > loss / val > valloss

#9-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#가장 무식한 방법 - 직접 하기
x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

# val 은 model.fit 에 적용이 된다.
x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#verbose를 추가하는 건 #3 컴파일, 훈련 단계에서 추가한다.
#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 100, batch_size = 4, 
           verbose = 1,
           validation_data = (x_val, y_val),
           )
# verbose = 0 : 침묵
# verbose = 1 : default 
# verbose = 2 : 프로그레스 바만 삭제
# verbose = 나머지 : epochs 횟수만 출력

#4.평가, 예측
loss = model.evaluate (x_test, y_test) 
print ('loss :', loss)
#validaion 로스는 훈련에 직접적으로 영향을 미친다 > No
#사람이 val loss 를 보고 파라미터를 조정한다 > 간접적으로 영향을 미친다. > Yes