# acc 1.0 만들기
"""
loss : 0.013718152418732643
acc : 0.9951456189155579
accuracy score : 1.0
걸린 시간 : 181.22 s
"""
import time
import datetime
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. 데이터 경로 설정 (horse-human 최상위 폴더 지정)
# ImageDataGenerator는 지정된 경로 아래의 하위 폴더(horses, humans)를 각각 클래스로 인식
path_data = "./_data/image/horse-human/"
path_save = "./_save/keras46_horses/"

xyall_dategen = ImageDataGenerator (
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

xy_all = xyall_dategen.flow_from_directory (
    path_data, # 경로
    target_size = (300, 300),
    batch_size= 2000, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'categorical',
    color_mode='rgb', # 컬러
    shuffle = True
)
# Found 160 images belonging to 2 classes.
# xy_test = test_dategen.flow_from_directory (
#     path_data,
#     target_size = (150, 150),
#     batch_size= 120, 
#     # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
#     class_mode= 'binary', # 이진분류라는 얘기
#     color_mode='grayscale', # 흑백
#     # shuffle = True # 테스트 데이터는 건들지 않는 것이 원칙이므로 shuffle을 할 필요가 없다.
# )
# # Found 120 images belonging to 2 classes.

# print (xy_all)
x = xy_all[0][0]
y = xy_all[0][1]
# print (x.shape) # (1027, 300, 300, 1)
# print (y.shape) # (1027,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.8, 
    random_state=333, 
    stratify=y # 클래스 비율 유지
)

# print (x_train.shape, y_train.shape) #(821, 300, 300, 1) (821,)
# print (x_test.shape, y_test.shape) #(206, 300, 300, 1) (206,)

np_path = "./_data/horse_npy/"
np.save(np_path + 'keras46_01_x_train.npy', arr=x_train) # x_train / arr=xy_train[0][0]도 가능
np.save(np_path + 'keras46_01_y_train.npy', arr=y_train) # y_train
np.save(np_path + 'keras46_01_x_test.npy', arr=x_test) # x_test
np.save(np_path + 'keras46_01_y_test.npy', arr=y_test) #y_test

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(300,300,3),padding='same', strides=2)) 
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
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam',
               metrics=['acc'])

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k46_", date, filename])

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