import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1.데이터 (.shape 확인)
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])

x = x.T
print (x)

# x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]]) # (5,2) <- 이제는 백터가 아니라 행렬
y = np.array([1,2,3,4,5]) # (5,)

print(x.shape) #(5, 2)
print(y.shape) #(5,)

#2.모델구성
model = Sequential() # 앞에 있는 문자가 대문자인 애는 통상적으로 클래스
#행렬에서 input_dim은 열의 개수와 동일하다.
#행 무시, 열 우선 (중요)
model.add(Dense(5, input_dim = 2)) 
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일 훈련 
model.compile (loss='mse', optimizer = 'adam')
model.fit (x, y, epochs = 100, batch_size = 3)
#  x  y
# 1 6 1
# 2 7 2
# 3 8 3
# 4 9 4
# 5 10 5
# 6 11 ?

#4.평가예측
loss = model.evaluate(x, y)
print ('loss = ',loss) # loss =  0.005197925493121147

results = model.predict(np.array([[6, 11]])) # 여기서는 [1,2]로 한거임
print ('[6, 11]의 예측값 : ', results) # [6, 11]의 예측값 :  [[5.8717375]]