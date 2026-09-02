#다시 해보기
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


#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 1000, batch_size = 5) #데이터보다 배치 사이즈가 크면 걍 통 배치로 돈다.

#4.평가, 예측
loss = model.evaluate (x_test, y_test) #evaulate를 이 테스트 셋들로 하겠다. 훈련과 관련이 없는 데이터들
print ('loss :', loss) #loss : 0.2941036820411682




