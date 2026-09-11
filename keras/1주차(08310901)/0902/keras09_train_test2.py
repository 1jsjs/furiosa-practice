import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 가장 무식한 방법 - 직접 하기
# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])


# 직접 하지말고 슬라이싱으로 하기
#[찾아보기] 넘파이 리스트의 슬라이싱으로 7대 3으로 나누기 - 리스트에서 하는 것과 동일
# https://076923.github.io/posts/Python-numpy-5/
# https://coddy.tech/learn/ko/courses/numpy_fundamentals/slicing

x_train = x[0:7]
y_train = y[:7] #0 제외해도 가능
print (x_train, y_train)

x_test = x[7:10]
y_test = y[7:] #0 제외해도 가능
print (x_test, y_test)

#keras09_train_test1.py에서 복사
#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))


#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 1000, batch_size = 5) #데이터보다 배치 사이즈가 크면 걍 통 배치로 돈다.

#4.평가, 예측
loss = model.evaluate (x_test, y_test) #evaulate를 이 테스트 셋들로 하겠다. 훈련과 관련이 없는 데이터들
print ('loss :', loss) #loss : 3.021118936885614e-05 = 0.00003021118936885614