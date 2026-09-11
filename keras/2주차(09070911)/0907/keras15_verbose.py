#9-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#가장 무식한 방법 - 직접 하기
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#verbose를 추가하는 건 #3 컴파일, 훈련 단계에서 추가한다.
#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 100, batch_size = 4, 
           verbose = 0,
           )
# verbose = 0 : 침묵
# verbose = 1 : default 
# verbose = 2 : 프로그레스 바만 삭제
# verbose = 나머지 : epochs 횟수만 출력

#4.평가, 예측
loss = model.evaluate (x_test, y_test) 
print ('loss :', loss) 