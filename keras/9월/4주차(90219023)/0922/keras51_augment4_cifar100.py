import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


path_save = "./_save/keras51_cifar100/"

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print (x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print (x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
# print (np.max(x_train), np.min(x_train)) #255 0
# print (np.max(x_test), np.min(x_test)) #255 0
# print (np.max(y_train), np.min(y_train)) #9 0
# print (np.max(y_test), np.min(y_test)) #9 0


############# 여기부터 증폭 #################
datagen = ImageDataGenerator (
    rescale = 1./255,
    # horizontal_flip = True, # 수평 뒤집기 (좌우반전)
    vertical_flip = True, # 수직 뒤집기 (상하반전)
    width_shift_range = 0.1, # 평행 이동
    # height_shift_range = 0.1,
    rotation_range = 15, # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range = 0.5, # 0이 default
    # shear_range = 0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest'
)

augment_size = 30000
randidx = np.random.randint(x_train.shape[0], size = augment_size) #5만개 중에 3만개 랜덤뽑기

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# 3-A. 증폭 데이터 생성 (rescale 적용됨)
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

# 3-B. 원본 데이터도 rescale 적용 (증폭 옵션 없는 별도 datagen 사용)
datagen_raw = ImageDataGenerator(rescale=1./255)
x_train = datagen_raw.flow(
    x_train, y_train,
    batch_size=x_train.shape[0],
    shuffle=False
).next()[0]

# 3-C. 테스트 데이터도 rescale 적용
x_test = datagen_raw.flow(
    x_test, y_test,
    batch_size=x_test.shape[0],
    shuffle=False
).next()[0]

# 4. 데이터 합치기 (모두 0~1 스케일)
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print("x_train min/max :", np.min(x_train), np.max(x_train)) #x_train min/max : 0.0 1.0


######## 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3), activation="relu", )) # 30,30,64
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu',)) # 28,28,64
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 23,23,32
model.add(MaxPool2D())
model.add(Dropout(0.2))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()
# 3. 컴파일, 훈련
model.compile (loss='sparse_categorical_crossentropy', optimizer='adam',
               metrics=['acc'])

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k51_", date, filename])

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
    patience= 30, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
    verbose = 1, 
)

start_time = time.time()
model.fit (x_train, y_train, epochs=5000, batch_size=512, verbose=1, validation_split=0.1, callbacks = [es, mcp])
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')