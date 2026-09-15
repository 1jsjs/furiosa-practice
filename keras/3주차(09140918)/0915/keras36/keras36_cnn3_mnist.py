import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
print (x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print (x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
print (np.max(x_train), np.min(x_train)) #255 0
print (np.max(x_test), np.min(x_test)) #255 0
print (np.max(y_train), np.min(y_train)) #9 0
print (np.max(y_test), np.min(y_test)) #9 0
"""
######## 1. 데이터
########## 1-1. 스케일링 1(대상 x)
############## ㄴ 0-1 사이인 minmax가 괜찮겠다. 근데 최솟값이 0이니  maxabs도 괜찮겠다. 지금은 민맥스나 맥스앱스나 같음
"""
x_train = x_train/255.
x_test = x_test/255.
print (np.max(x_train), np.min(x_train)) #1.0 0.0
print (np.max(x_test), np.min(x_test)) #1.0 0.0
"""

########## 1-2. 스케일링 2(대상 x)
############## 이미지에서 한정, -1 ~ +1 로 한다. -> 이미지 쪽 전처리는 이렇게 많이 한다.
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print (np.max(x_train), np.min(x_train)) #1.0 -1.0
print (np.max(x_test), np.min(x_test)) #1.0 -1.0