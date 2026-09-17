import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# 2. 모델구성
model = Sequential()
model.add (Conv2D(filters=10, kernel_size= (2,2), input_shape=(10,10,1),
                  padding='same', strides=2, #padding은 valid 가 default.
                  ))
model.add(Conv2D(filters=9, kernel_size=(3,3),
                  padding='same', strides=1, #padding은 valid 가 default.
                 ))

model.summary()