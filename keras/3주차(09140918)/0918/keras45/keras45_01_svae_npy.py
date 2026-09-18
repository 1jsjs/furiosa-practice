# 44-2 copy

"""

"""

import time
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# 수치화 + 데이터 증폭 => ImageDataGemerator

path_train = "./_data/image/brain/train"
path_test = "./_data/image/brain/test"

train_dategen = ImageDataGenerator (
    rescale = 1./255,
    # horizontal_flip = True, # 수평 뒤집기
    # vertical_flip = True, # 수직 뒤집기 (상하반전)
    # width_shift_range = 0.1, # 평행 이동
    # height_shift_range = 0.1,
    # rotation_range = 5, # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.2,
    # shear_range = 0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (한마디로 피부...?)
    # fill_mode='nearest'
)
test_dategen = ImageDataGenerator (
    rescale = 1./255,
)

xy_train = train_dategen.flow_from_directory (
    path_train, # 경로
    target_size = (150, 150),
    batch_size= 160, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'binary', # 이진분류라는 얘기
    color_mode='grayscale', # 흑백
    shuffle = True
)
# Found 160 images belonging to 2 classes.
xy_test = test_dategen.flow_from_directory (
    path_test,
    target_size = (150, 150),
    batch_size= 120, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'binary', # 이진분류라는 얘기
    color_mode='grayscale', # 흑백
    # shuffle = True # 테스트 데이터는 건들지 않는 것이 원칙이므로 shuffle을 할 필요가 없다.
)
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

print (x_train.shape, y_train.shape) #(160, 150, 150, 1) (160,)
print (x_test.shape, y_test.shape) #(120, 150, 150, 1) (120,)


np_path = "./_data/brain_npy/"
np.save(np_path + 'keras45_01_x_train.npy', arr=x_train) # x_train / arr=xy_train[0][0]도 가능
np.save(np_path + 'keras45_01_y_train.npy', arr=y_train) # y_train
np.save(np_path + 'keras45_01_x_test.npy', arr=x_test) # x_test
np.save(np_path + 'keras45_01_y_test.npy', arr=y_test) #y_test

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(150,150,1),padding='same', strides=2)) 
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu',padding='same', strides=2)) 
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', strides=2))
model.add(MaxPool2D())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', strides=2)) 
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=8, kernel_size=(2,2), activation='relu', padding='same', strides=2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류는 sigmoid !!!!!!!!

# 3. 컴파일, 훈련
model.compile (loss='binary_crossentropy', optimizer='adam', #이진분류!!!!
               metrics=['acc'])
start_time = time.time()
model.fit (x_train, y_train, epochs=500,
           batch_size=8, verbose=1, validation_split=0.1)
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)

y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', round(acc_score, 2))
print ('걸린 시간 :', round(end_time - start_time, 2), 's')