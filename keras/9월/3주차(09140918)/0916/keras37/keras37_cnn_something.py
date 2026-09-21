# 36-1 copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, # 커널 개수
                 (3,3), # 각 커널의 크기 
                 input_shape=(10,10,1))) # (height, width, channel)
model.add(Conv2D(5, #커널 개수
                 (2,2))) # 각 커널의 크기 

model.summary()
