import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1.데이터 (.shape 확인)
xWrong = np.array([[1,2,3,4,5,6], 
               [6,7,8,9,10]])
#(2,5)이라 y랑 shape 안 맞음
x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]]) # (5,2)
y = np.array([1,2,3,4,5]) # (5,)

print(xWrong.shape)
print(x.shape)
print(y.shape)


#2.모델구성

#3.컴파일 훈련 

#4.평가예측
