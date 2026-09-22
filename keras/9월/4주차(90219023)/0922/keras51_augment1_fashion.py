# 50-2 copy
"""
Keras의 ImageDataGenerator에서 flow() 메소드는 메모리에 저장된 단일 이미지 또는 이미지 배열(NumPy array)을 입력받아 
지정된 증폭(Data Augmentation) 조건을 적용한 뒤, 배치(batch) 단위로 무한히 생성해내는 데이터 이터레이터(Iterator)를 반환하는 역할을 합니다.
"""
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

path_save = "./_save/keras51_fashion_mnist/"



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

augment_size = 40000
# print (x_train.shape[0]) #60000
randidx = np.random.randint(x_train.shape[0], size = augment_size) #6만개 중에 4만개 랜덤뽑기

# print (randidx.shape) #벡터니까 shape 된다. // 리스트와 튜플은 len 명령어로 확인해야 한다.
# print (np.min(randidx), np.max(randidx)) #랜덤값임

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print (x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)
x_augmented = x_augmented.reshape (
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],
    1
)
# print (x_augmented.shape) #(40000, 28, 28, 1)


xy_augmented = datagen.flow (
                            x_augmented, y_augmented,
                            batch_size = augment_size,
                            shuffle = False,
).next()[0]

############# 변환 완료 ##############
# print (x_augmented.shape) #(40000, 28, 28, 1)

# print (x_train.shape) #(60000, 28, 28)
x_train = x_train.reshape (60000,28,28,1)
x_test = x_test.reshape (10000,28,28,1)

x_train = np.concatenate ((x_train, x_augmented))
y_train = np.concatenate ((y_train, y_augmented))

# print (x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train) 
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)
print (y_train.shape, y_test.shape)

######## 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(28,28,1),)) # 26=(28-3+1),26=(28-3+1),64 가 됨
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu',)) #24=(26-3+1),24=(26-3+1),32
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 23=(24-2+1),23=(24-2+1),32
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 22,22,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # 21,21,16
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(10, activation='softmax')) #원하는 shape상태는 (10,)이다.

# 3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam',
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
    patience= 100, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
    verbose = 1, 
)

start_time = time.time()
model.fit (x_train, y_train, epochs=10000, batch_size=128, verbose=1, validation_split=0.1, callbacks=[es, mcp])
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred = np.argmax(y_pred, axis=1) #.reshape(-1,1)
y_test = np.argmax(y_test, axis=1) #.reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', acc_score)
print ('걸린 시간 :', round(end_time - start_time, 2), 's')

"""
loss : 0.33616891503334045
acc : 0.9117000102996826
accuracy score : 0.9117
걸린 시간 : 6374.61 s
"""