# https://www.kaggle.com/datasets/tongpython/cat-and-dog/data
# 44-3 copy
"""

"""
#cat-and-dog

import time
import datetime
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# 수치화 + 데이터 증폭 => ImageDataGemerator

path_train = "./_data/image/catdog/training_set"
path_test = "./_data/image/catdog/test_set"
path_save = "./_save/keras44_CatDog/"

"""
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
    batch_size= 10000, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'binary', # 이진분류라는 얘기
    color_mode= "rgb", # 컬러
    shuffle = True
)
# Found 8005 images belonging to 2 classes.
xy_test = test_dategen.flow_from_directory (
    path_test,
    target_size = (150, 150),
    batch_size= 10000, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'binary', # 이진분류라는 얘기
    color_mode= "rgb", # 흑백
    # shuffle = True # 테스트 데이터는 건들지 않는 것이 원칙이므로 shuffle을 할 필요가 없다.
)
# Found 2023 images belonging to 2 classes

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print (x_train.shape, y_train.shape) #(160, 150, 150, 1) (160,)
# print (x_test.shape, y_test.shape) #(120, 150, 150, 1) (120,)
"""

np_path = "./_data/kaggle_cat_dog_npy/"

x_train = np.load (np_path + 'keras45_01_x_train.npy')
y_train = np.load (np_path + 'keras45_01_y_train.npy')
x_test = np.load (np_path + 'keras45_01_x_test.npy')
y_test = np.load (np_path + 'keras45_01_y_test.npy')


# 2. 모델 구성
model = Sequential()

model.add(Conv2D(64, (3,3), activation="relu", input_shape=(300, 300, 3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(512, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

# 분류기 (Classifier) 부분
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile (loss='binary_crossentropy', optimizer='adam', #이진분류!!!!
               metrics=['acc'])
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k44_", date, filename])
# 파일명 예
# "./_save/keras30/" + "k30_", 0914_1147, 에포수-val_loss의 소수 4번째자리까지.keras
mcp = ModelCheckpoint ( 
    monitor='val_loss',
    mode='auto',
    save_best_only = True,
    filepath =filepath,
    verbose=1,
)

es = EarlyStopping (
    monitor= 'val_loss', #기준을 선언
    mode= 'min', #어떤 값을 찾을까? 긴가민가 하면 auto 하면 됨
    patience= 100, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
    verbose = 1, 
)

start_time = time.time()
model.fit (x_train, y_train, epochs=5000,
           batch_size=4, verbose=1, validation_split=0.1, callbacks=[es, mcp])
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