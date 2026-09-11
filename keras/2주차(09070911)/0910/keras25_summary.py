import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#2. 모델 구성
model = Sequential()
model.add (Dense(3, input_dim = 1))
model.add (Dense(4))
model.add (Dense(3))
model.add (Dense(1))
# 총 파라미터 30개? 1*3 + 3*4 + 3*4 + 3*1
model.summary() #  Total params: 41 > 바이어스도 포함되기 때문
