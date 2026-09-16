# 38-1 copy

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D #MaxPooling2D

# 2. 모델구성
model = Sequential()
model.add (Conv2D(filters=10, kernel_size= (2,2), input_shape=(10,10,1),
                  padding='same', strides=1, #padding은 valid 가 default.
                  ))
model.add(MaxPool2D())
model.add(Conv2D(filters=9, kernel_size=(3,3),
                  padding='same', strides=2, #padding은 valid 가 default.
                 ))

model.summary()
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 10, 10, 10)        50        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 5, 5, 10)         0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 3, 3, 9)           819       
                                                                 
=================================================================
Total params: 869
Trainable params: 869
Non-trainable params: 0
_________________________________________________________________
"""